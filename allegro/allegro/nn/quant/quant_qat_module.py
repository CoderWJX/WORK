from typing import List, Optional

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode


from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from allegro.nn import ScalarMLPFunction


@compile_mode("script")
class QuantScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
            self,
            mlp_latent_dimensions: List[int],
            mlp_output_dimension: Optional[int],
            mlp_nonlinearity: Optional[str] = "silu",
            mlp_initialization: str = "uniform",
            mlp_dropout_p: float = 0.0,
            mlp_batchnorm: bool = False,
            field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: Optional[str] = None,
            irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1)  # scalars
        in_dim = self.irreps_in[self.field][0].mul
        self._module = QuantScalarMLPFunction(
            mlp_input_dimension=in_dim,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension,
            mlp_nonlinearity=mlp_nonlinearity,
            mlp_initialization=mlp_initialization,
            mlp_dropout_p=mlp_dropout_p,
            mlp_batchnorm=mlp_batchnorm,
        )
        self.irreps_out[self.out_field] = o3.Irreps(
            [(self._module.out_features, (0, 1))]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self._module(data[self.field])
        return data


class QuantScalarMLPFunction(ScalarMLPFunction):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
            self,
            m: ScalarMLPFunction,
            quan_w_fn: None,
            quan_a_fn: None,
    ):
        assert type(m) == ScalarMLPFunction
        super().__init__(m.mlp_input_dimension,
                         m.mlp_latent_dimensions,
                         m.mlp_output_dimension,
                         m.mlp_nonlinearity,
                         m.mlp_initialization,
                         m.mlp_dropout_p,
                         m.mlp_batchnorm,)

        # quant functions add by yujie.zeng@2023.01.31
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.register_buffer('t', torch.ones(1))

        dimensions = (
                ([m.mlp_input_dimension] if m.mlp_input_dimension is not None else [])
                + m.mlp_latent_dimensions
                + ([m.mlp_output_dimension] if m.mlp_output_dimension is not None else [])
        )

        # Code
        params = {}
        modules = {}
        qt_graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(qt_graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            exec('self.w_{} = self.graph.owning_module._weight_{}.detach()'.format(layer, layer))
            exec('params[f"_weight_{}"] = self.w_{}'.format(layer, layer))

        modules[f"quan_w_fn"] = self.quan_w_fn
        modules[f"quan_a_fn"] = self.quan_a_fn

        first_matmul = True
        env = {}
        quant_w = True
        quant_w_t = None
        # generate code
        for node in list(self.graph.nodes):
            new_node = qt_graph.node_copy(node, lambda x: env[x.name])
            if quant_w:
                if node.op == "call_function" and node.target == torch.matmul and first_matmul:
                    new_node.update_arg(1, quant_w_t)
                    first_matmul = False
                elif new_node.op == "call_function" and "mul" in node.name and first_matmul:
                    with qt_graph.inserting_after(new_node):
                        quant_w_t = qt_graph.call_module(f"quan_w_fn", (new_node,))

            env[node.name] = new_node

        qt_graph.lint()

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        for mname, m in modules.items():
            setattr(base, mname, m)

        self._codegen_register({"_qt_forward": fx.GraphModule(base, qt_graph)})

    def forward(self, x):
        quantized_act = self.quan_a_fn(x)
        res = self._qt_forward(quantized_act * self.t)
        return res