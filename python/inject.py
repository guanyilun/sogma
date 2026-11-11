"""Signal injection for mapmaking tests (yilung-fixes, 2026-11-11).  When SOGMA_INJECT names a T,Q,U enmap, every observation's time stream gets that map's
signal added (SOGMA_INJECT_MODE=add, default) or substituted (replace) before deslope, gapfill and the noise model,
evaluated with the exact pointing of calc_pointing in SOGMA_INJECT_SYS (e.g. sidelobe,on=Sun) and the formula of
map2tod: d = r_T T + r_P (Q cos 2psi + U sin 2psi), bilinear in the map."""
import os
import numpy as np
from pixell import enmap
from . import pmat
from .logging import L

_MAP = None


def active():
	return bool(os.environ.get("SOGMA_INJECT"))


def apply(gtod, obs, dev, chunk=64):
	global _MAP
	if _MAP is None:
		_MAP = enmap.read_map(os.environ["SOGMA_INJECT"])
	sys  = os.environ["SOGMA_INJECT_SYS"]
	mode = os.environ.get("SOGMA_INJECT_MODE", "add")
	resp = np.asarray(obs.response) if getattr(obs, "response", None) is not None else None
	ndet = gtod.shape[0]
	rms = 0.0
	for d0 in range(0, ndet, chunk):
		d1  = min(ndet, d0 + chunk)
		pos = pmat.calc_pointing(obs.ctime, obs.boresight, obs.point_offset[d0:d1], obs.polangle[d0:d1],
			sys=sys, site=obs.site, dtype=np.float64)                     # [{dec,ra,psi},ndet,nsamp]
		val = _MAP.at(pos[:2], order=1, border="wrap")                         # [3,ndet,nsamp]
		rT  = resp[0, d0:d1, None] if resp is not None else 1.0
		rP  = resp[1, d0:d1, None] if resp is not None else 1.0
		sig = rT * val[0] + rP * (val[1] * np.cos(2 * pos[2]) + val[2] * np.sin(2 * pos[2]))
		sig = dev.np.asarray(sig.astype(gtod.dtype))
		if mode == "replace": gtod[d0:d1] = sig
		else:                 gtod[d0:d1] += sig
		rms += float(np.sum(sig.astype(np.float64) ** 2)) if mode else 0.0
	L.print("Injected %s (%s) rms %.4g" % (os.path.basename(os.environ["SOGMA_INJECT"]), mode, (rms / gtod.size) ** 0.5), level=2)
