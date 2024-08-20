class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_93.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_94.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_95.Conv,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv = self.conv
    _0 = (act).forward((conv).forward(argument_1, ), )
    return _0
