class Model(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  model : __torch__.torch.nn.modules.container.___torch_mangle_199.Sequential
  def forward(self: __torch__.models.yolo.Model,
    x: Tensor) -> Tuple[Tensor]:
    model = self.model
    _24 = getattr(model, "24")
    model0 = self.model
    _23 = getattr(model0, "23")
    model1 = self.model
    _22 = getattr(model1, "22")
    model2 = self.model
    _21 = getattr(model2, "21")
    model3 = self.model
    _20 = getattr(model3, "20")
    model4 = self.model
    _19 = getattr(model4, "19")
    model5 = self.model
    _18 = getattr(model5, "18")
    model6 = self.model
    _17 = getattr(model6, "17")
    model7 = self.model
    _16 = getattr(model7, "16")
    model8 = self.model
    _15 = getattr(model8, "15")
    model9 = self.model
    _14 = getattr(model9, "14")
    model10 = self.model
    _13 = getattr(model10, "13")
    model11 = self.model
    _12 = getattr(model11, "12")
    model12 = self.model
    _11 = getattr(model12, "11")
    model13 = self.model
    _10 = getattr(model13, "10")
    model14 = self.model
    _9 = getattr(model14, "9")
    model15 = self.model
    _8 = getattr(model15, "8")
    model16 = self.model
    _7 = getattr(model16, "7")
    model17 = self.model
    _6 = getattr(model17, "6")
    model18 = self.model
    _5 = getattr(model18, "5")
    model19 = self.model
    _4 = getattr(model19, "4")
    model20 = self.model
    _3 = getattr(model20, "3")
    model21 = self.model
    _2 = getattr(model21, "2")
    model22 = self.model
    _1 = getattr(model22, "1")
    model23 = self.model
    _0 = getattr(model23, "0")
    _25 = (_2).forward((_1).forward((_0).forward(x, ), ), )
    _26 = (_4).forward((_3).forward(_25, ), )
    _27 = (_6).forward((_5).forward(_26, ), )
    _28 = (_9).forward((_8).forward((_7).forward(_27, ), ), )
    _29 = (_10).forward(_28, )
    _30 = (_12).forward((_11).forward(_29, ), _27, )
    _31 = (_14).forward((_13).forward(_30, ), )
    _32 = (_16).forward((_15).forward(_31, ), _26, )
    _33 = (_17).forward(_32, )
    _34 = (_19).forward((_18).forward(_33, ), _31, )
    _35 = (_20).forward(_34, )
    _36 = (_22).forward((_21).forward(_35, ), _29, )
    _37 = (_24).forward(_33, _35, (_23).forward(_36, ), )
    return (_37,)
class Detect(Module):
  __parameters__ = []
  __buffers__ = ["anchors", ]
  anchors : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  m : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.models.yolo.Detect,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor) -> Tensor:
    m = self.m
    _2 = getattr(m, "2")
    m0 = self.m
    _1 = getattr(m0, "1")
    m1 = self.m
    _0 = getattr(m1, "0")
    _13 = (_0).forward(argument_1, )
    bs = ops.prim.NumToTensor(torch.size(_13, 0))
    _14 = int(bs)
    _15 = int(bs)
    ny = ops.prim.NumToTensor(torch.size(_13, 2))
    _16 = int(ny)
    nx = ops.prim.NumToTensor(torch.size(_13, 3))
    _17 = torch.view(_13, [_15, 3, 25, _16, int(nx)])
    _18 = torch.contiguous(torch.permute(_17, [0, 1, 3, 4, 2]))
    _19 = torch.split_with_sizes(torch.sigmoid(_18), [2, 2, 21], 4)
    xy, wh, conf, = _19
    _20 = torch.add(torch.mul(xy, CONSTANTS.c0), CONSTANTS.c1)
    xy0 = torch.mul(_20, torch.select(CONSTANTS.c2, 0, 0))
    _21 = torch.pow(torch.mul(wh, CONSTANTS.c0), 2)
    wh0 = torch.mul(_21, CONSTANTS.c3)
    y = torch.cat([xy0, wh0, conf], 4)
    _22 = torch.view(y, [_14, -1, 25])
    _23 = (_1).forward(argument_2, )
    bs0 = ops.prim.NumToTensor(torch.size(_23, 0))
    _24 = int(bs0)
    _25 = int(bs0)
    ny0 = ops.prim.NumToTensor(torch.size(_23, 2))
    _26 = int(ny0)
    nx0 = ops.prim.NumToTensor(torch.size(_23, 3))
    _27 = torch.view(_23, [_25, 3, 25, _26, int(nx0)])
    _28 = torch.contiguous(torch.permute(_27, [0, 1, 3, 4, 2]))
    _29 = torch.split_with_sizes(torch.sigmoid(_28), [2, 2, 21], 4)
    xy1, wh1, conf0, = _29
    _30 = torch.add(torch.mul(xy1, CONSTANTS.c0), CONSTANTS.c4)
    xy2 = torch.mul(_30, torch.select(CONSTANTS.c2, 0, 1))
    _31 = torch.pow(torch.mul(wh1, CONSTANTS.c0), 2)
    wh2 = torch.mul(_31, CONSTANTS.c5)
    y0 = torch.cat([xy2, wh2, conf0], 4)
    _32 = torch.view(y0, [_24, -1, 25])
    _33 = (_2).forward(argument_3, )
    bs1 = ops.prim.NumToTensor(torch.size(_33, 0))
    _34 = int(bs1)
    _35 = int(bs1)
    ny1 = ops.prim.NumToTensor(torch.size(_33, 2))
    _36 = int(ny1)
    nx1 = ops.prim.NumToTensor(torch.size(_33, 3))
    _37 = torch.view(_33, [_35, 3, 25, _36, int(nx1)])
    _38 = torch.contiguous(torch.permute(_37, [0, 1, 3, 4, 2]))
    _39 = torch.split_with_sizes(torch.sigmoid(_38), [2, 2, 21], 4)
    xy3, wh3, conf1, = _39
    _40 = torch.add(torch.mul(xy3, CONSTANTS.c0), CONSTANTS.c6)
    xy4 = torch.mul(_40, torch.select(CONSTANTS.c2, 0, 2))
    _41 = torch.pow(torch.mul(wh3, CONSTANTS.c0), 2)
    wh4 = torch.mul(_41, CONSTANTS.c7)
    y1 = torch.cat([xy4, wh4, conf1], 4)
    _42 = [_22, _32, torch.view(y1, [_34, -1, 25])]
    return torch.cat(_42, 1)
