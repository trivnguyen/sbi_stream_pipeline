"""Target observational data: load from a survey catalog once, then keep a
minimal, self-contained copy inside the run directory."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class TargetData:
    """Self-contained snapshot of one target's observational data.

    Attributes:
        phi1: Stream-aligned longitude (deg)
        phi2: Stream-aligned latitude (deg)
        vr: Heliocentric radial velocity (km/s)
        vr_err: Radial velocity uncertainty (km/s)
        pm1: Proper motion along phi1 (mas/yr)
        pm1_err: Proper motion uncertainty along phi1 (mas/yr)
        pm2: Proper motion along phi2 (mas/yr)
        pm2_err: Proper motion uncertainty along phi2 (mas/yr)
        dist: Heliocentric distance (kpc)
        dist_err: Distance uncertainty (kpc)
    """
    key: str
    phi1: np.ndarray
    phi2: np.ndarray
    vr: np.ndarray
    vr_err: np.ndarray
    pm1: np.ndarray
    pm1_err: np.ndarray
    pm2: np.ndarray
    pm2_err: np.ndarray
    dist: np.ndarray
    dist_err: np.ndarray

    @classmethod
    def from_catalog(cls, target_config) -> 'TargetData':
        """Load target data from a survey catalog via `dsph_analysis`.

        Args:
            target_config: ConfigDict with `catalog_path` and `key` fields.

        Returns:
            A self-contained TargetData snapshot.
        """
        df = pd.read_csv(target_config.catalog_path)
        return cls(
            key=target_config.key,
            phi1=df['phi1'].to_numpy(),
            phi2=df['phi2'].to_numpy(),
            vr=df['vr'].to_numpy(),
            vr_err=df['vr_err'].to_numpy(),
            pm1=df['pm1'].to_numpy(),
            pm1_err=df['pm1_err'].to_numpy(),
            pm2=df['pm2'].to_numpy(),
            pm2_err=df['pm2_err'].to_numpy(),
            dist=df['dist'].to_numpy(),
            dist_err=df['dist_err'].to_numpy()
        )

    def save(self, path) -> None:
        """Save this snapshot to a single .npz file.

        Args:
            path: Destination .npz path.
        """
        np.savez(
            path, key=self.key,
            phi1=self.phi1, phi2=self.phi2,
            vr=self.vr, vr_err=self.vr_err,
            pm1=self.pm1, pm1_err=self.pm1_err,
            pm2=self.pm2, pm2_err=self.pm2_err,
            dist=self.dist, dist_err=self.dist_err,
        )

    @classmethod
    def load(cls, path) -> 'TargetData':
        """Load a snapshot previously written by `save`.

        Args:
            path: Source .npz path.

        Returns:
            The loaded TargetData snapshot.
        """
        raw = np.load(path)
        return cls(
            key=raw['key'].item(),
            phi1=raw['phi1'], phi2=raw['phi2'],
            vr=raw['vr'], vr_err=raw['vr_err'],
            pm1=raw['pm1'], pm1_err=raw['pm1_err'],
            pm2=raw['pm2'], pm2_err=raw['pm2_err'],
            dist=raw['dist'], dist_err=raw['dist_err'],
        )
