"""
本地 CPU 小网格自检用：把 cupy / cupyx 换成 numpy / scipy 的同名函数，好让 topopt3d_noinfill 的 Pipeline、MMA 在 CPU 上跑。
只在 link3d.py --cpu 时装上；GPU kernel（RawModule）不能用，用到就报错。
"""
import sys
import types
import numpy as np
import scipy.ndimage
import scipy.sparse


class _CupyShim(types.ModuleType):
    def __getattr__(self, name):
        return getattr(np, name)


def _no_kernel(*a, **k):
    raise RuntimeError("CPU 自检模式下不能用 GPU kernel")


def install():
    if isinstance(sys.modules.get("cupy"), _CupyShim):
        return
    cp = _CupyShim("cupy")
    cp.asnumpy = np.asarray
    cp.ElementwiseKernel = lambda *a, **k: None
    cp.RawModule = _no_kernel
    cp.ndarray = np.ndarray
    cp.bool_ = np.bool_
    cupyx = types.ModuleType("cupyx")
    cxs = types.ModuleType("cupyx.scipy")
    cxs.ndimage = scipy.ndimage
    cxs.sparse = scipy.sparse
    cupyx.scipy = cxs
    sys.modules.update({"cupy": cp, "cupyx": cupyx, "cupyx.scipy": cxs, "cupyx.scipy.ndimage": scipy.ndimage,
                        "cupyx.scipy.sparse": scipy.sparse})
