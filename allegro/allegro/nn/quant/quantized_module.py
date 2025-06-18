import torch
from .._fc import ScalarMLPFunction
from .quant_qat_module import QuantScalarMLPFunction
from .lsq import grad_scale, round_pass

class ReScale(torch.nn.Module):
    def __init__(self, act_scale, wei_scale, out_type='int8'):
        super(ReScale, self).__init__()
        self.wei_scale = torch.nn.Parameter(wei_scale)
        self.act_scale = torch.nn.Parameter(act_scale)

        if out_type == 'uint8':
            self.thd_neg = 0
            self.thd_pos = 255
        elif out_type == 'int8':
            self.thd_neg = -128
            self.thd_pos = 127
        elif out_type == 'int16':
            self.thd_neg = -32768
            self.thd_pos = 32767
        elif out_type == 'uint16':
            self.thd_neg = 0
            self.thd_pos = 65535
        else:
            raise NotImplementedError

    def forward(self, x, a_scale):
        return torch.clamp(x * a_scale * self.wei_scale, self.thd_neg, self.thd_pos)

    def quant(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.act_scale, torch.tensor(s_grad_scale))

        x = x / s_scale
        x = torch.clamp(torch.nan_to_num(x), self.thd_neg, self.thd_pos)
        x = round_pass(x)
        return x, s_scale

class QuantizedScalarMLPFunction(torch.nn.Module):
    def __init__(self, m: QuantScalarMLPFunction, act_scale, wei_scale, name, out_type='int8', input_type='int8'):
        super(QuantizedScalarMLPFunction, self).__init__()
        self.fc = ReScalarMLPFunction(m, act_scale, wei_scale, name)
        self.input_type = input_type

    def forward(self, x):
        x, scale = self.fc.Rescale.quant(x)
        x = self.fc(x, scale)
        return x

class ReScalarMLPFunction(ScalarMLPFunction):
    """Module implementing an MLP according to provided options."""
    in_features: int
    out_features: int

    def __init__(
            self,
            m: QuantScalarMLPFunction,
            act_scale: None,
            wei_scale: None,
            name
    ):
        super().__init__(
            m.mlp_input_dimension,
            m.mlp_latent_dimensions,
            m.mlp_output_dimension,
            m.mlp_nonlinearity,
            m.mlp_initialization,
            m.mlp_dropout_p,
            m.mlp_batchnorm,
        )
        self.Rescale = ReScale(act_scale, wei_scale)

        dimensions = (
                ([self.mlp_input_dimension] if self.mlp_input_dimension is not None else [])
                + self.mlp_latent_dimensions
                + ([self.mlp_output_dimension] if self.mlp_output_dimension is not None else [])
        )

        # Code
        params = {}
        modules = {}
        re_graph = torch.fx.Graph()
        tracer = torch.fx.proxy.GraphAppendingTracer(re_graph)

        def Proxy(n):
            return torch.fx.Proxy(n, tracer=tracer)

        base = torch.nn.Module()
        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            exec('self.w_{} = m.w_{}.detach()'.format(layer, layer))
            exec('params[f"_weight_{}"] = self.w_{}'.format(layer, layer))
        modules[f"Rescale"] = self.Rescale
        a_scale = Proxy(re_graph.placeholder("a_scale"))
        env = {}
        rescale = None
        # generate code
        layer_except = {'model.func.allegro.latents.0', 'model.func.edge_eng._module'}
        for node in list(self.graph.nodes):
            if node.target == 'output' and name not in layer_except:
                continue
            if node.name != "mul":
                new_node = re_graph.node_copy(node, lambda x: env[x.name])
            if node.name == 'matmul_1' and name == 'model.func.edge_eng._module':
                new_node.update_arg(0, rescale)
            if node.op == "call_function" and node.name == 'silu':
                new_node.update_arg(0, rescale)
            else:
                if new_node.op == "call_function" and node.name == 'matmul':
                    with re_graph.inserting_after(new_node):
                        rescale = re_graph.call_module(f"Rescale", (new_node, a_scale.node, ))
            env[node.name] = new_node
        re_graph.output(rescale)
        re_graph.lint()

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        for mname, m in modules.items():
            setattr(base, mname, m)
        self._codegen_register({"_rescale_forward": torch.fx.GraphModule(base, re_graph)})

    def forward(self, x, a_scale):
        return self._rescale_forward(a_scale, x)