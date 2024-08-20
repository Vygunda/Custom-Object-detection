class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_51.Conv
  cv2 : __torch__.models.common.___torch_mangle_54.Conv
  cv3 : __torch__.models.common.___torch_mangle_57.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_79.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_80.C3,
    argument_1: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _0 = (m).forward((cv1).forward(argument_1, ), )
    input = torch.cat([_0, (cv2).forward(argument_1, )], 1)
    return (cv3).forward(input, )
