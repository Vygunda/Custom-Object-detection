class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  act : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.models.common.Conv,
    x: Tensor) -> Tensor:
    act = self.act
    conv = self.conv
    _0 = (act).forward((conv).forward(x, ), )
    return _0
class C3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_5.Conv
  cv2 : __torch__.models.common.___torch_mangle_8.Conv
  cv3 : __torch__.models.common.___torch_mangle_11.Conv
  m : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.models.common.C3,
    argument_1: Tensor) -> Tensor:
    cv3 = self.cv3
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _1 = (m).forward((cv1).forward(argument_1, ), )
    input = torch.cat([_1, (cv2).forward(argument_1, )], 1)
    return (cv3).forward(input, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_14.Conv
  cv2 : __torch__.models.common.___torch_mangle_17.Conv
  def forward(self: __torch__.models.common.Bottleneck,
    argument_1: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _2 = (cv2).forward((cv1).forward(argument_1, ), )
    return torch.add(argument_1, _2)
class SPPF(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.models.common.___torch_mangle_104.Conv
  cv2 : __torch__.models.common.___torch_mangle_107.Conv
  m : __torch__.torch.nn.modules.pooling.MaxPool2d
  def forward(self: __torch__.models.common.SPPF,
    argument_1: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    cv1 = self.cv1
    _3 = (cv1).forward(argument_1, )
    _4 = (m).forward(_3, )
    _5 = (m).forward1(_4, )
    input = torch.cat([_3, _4, _5, (m).forward2(_5, )], 1)
    return (cv2).forward(input, )
class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.models.common.Concat,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    input = torch.cat([argument_1, argument_2], 1)
    return input
