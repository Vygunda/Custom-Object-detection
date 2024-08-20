class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.common.___torch_mangle_36.Bottleneck
  __annotations__["1"] = __torch__.models.common.___torch_mangle_43.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_44.Sequential,
    argument_1: Tensor) -> Tensor:
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _2 = (_1).forward((_0).forward(argument_1, ), )
    return _2
