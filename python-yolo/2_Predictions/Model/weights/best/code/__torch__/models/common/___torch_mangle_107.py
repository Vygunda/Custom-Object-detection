class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_105.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_106.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_107.Conv,
    input: Tensor) -> Tensor:
    act = self.act
    conv = self.conv
    _0 = (act).forward((conv).forward(input, ), )
    return _0
