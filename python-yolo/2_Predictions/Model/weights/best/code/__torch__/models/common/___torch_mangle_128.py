class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_113.Conv
  cv2 : __torch__.models.common.___torch_mangle_116.Conv
  cv3 : __torch__.models.common.___torch_mangle_119.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_127.Sequential
  def forward(self: __torch__.models.common.___torch_mangle_128.C3,
    argument_1: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _0 = (m).forward((cv1).forward(argument_1, ), )
    input = torch.cat([_0, (cv2).forward(argument_1, )], 1)
    return (cv3).forward(input, )
