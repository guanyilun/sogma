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


# Coverage diagnostic (yilung-fixes, 2026-11-12): when
# SOGMA_COVREF names a reference ivar map (e.g. another pipeline's), each observation logs how many of its uncut
# samples (every 8th detector, every 10th sample) land where the reference has no weight, and when (seconds from the
# start) and at what boresight elevation and azimuth they occur.  Off by default; changes nothing in the map.
_REF = None


def coverage_active():
	return bool(os.environ.get("SOGMA_COVREF"))


def coverage(id, obs, dstep=8, sstep=10):
	global _REF
	if _REF is None:
		r = enmap.read_map(os.environ["SOGMA_COVREF"])
		_REF = r if r.ndim == 2 else r[0]
	sys = os.environ.get("SOGMA_INJECT_SYS", "sidelobe,on=Sun")
	dets = np.arange(0, len(obs.point_offset), dstep)
	samps = np.arange(0, len(obs.ctime), sstep)
	cut = np.zeros((len(dets), len(samps)), bool)
	c = obs.cuts
	row = {d: i for i, d in enumerate(dets)}
	for d, s0, n in zip(np.asarray(c.dets), np.asarray(c.starts), np.asarray(c.lens)):
		if d in row:
			cut[row[d], np.searchsorted(samps, s0):np.searchsorted(samps, s0 + n)] = True
	bore = obs.boresight[:, samps] if obs.boresight.ndim == 2 else obs.boresight
	pos = pmat.calc_pointing(obs.ctime[samps], bore, obs.point_offset[dets], obs.polangle[dets],
		sys=sys, site=obs.site, dtype=np.float64)
	pix = enmap.sky2pix(_REF.shape, _REF.wcs, pos[:2]).astype(int)
	pix[1] %= _REF.shape[1]
	ok = (~cut) & (pix[0] >= 0) & (pix[0] < _REF.shape[0])
	out = ok & (_REF[np.clip(pix[0], 0, _REF.shape[0] - 1), pix[1]] <= 0)
	n, no = int(ok.sum()), int(out.sum())
	msg = "Coverage %s uncut %d outside-ref %d (%.4f)" % (id, n, no, no / max(n, 1))
	if no:
		t = obs.ctime[samps] - obs.ctime[0]
		js = np.nonzero(out.any(0))[0]
		b = np.degrees(np.asarray(bore))                 # rows as obs.boresight: el, az, roll
		msg += " t %.0f-%.0f s of %.0f; dets %d of %d; bore el %.2f-%.2f az %.1f-%.1f (all el %.2f-%.2f az %.1f-%.1f)" % (
			t[js[0]], t[js[-1]], t[-1], int(out.any(1).sum()), len(dets),
			b[0, js].min(), b[0, js].max(), b[1, js].min(), b[1, js].max(), b[0].min(), b[0].max(), b[1].min(), b[1].max())
	L.print(msg, level=1)
