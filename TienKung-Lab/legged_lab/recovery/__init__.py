"""LIPM/DCM recoverability tools (not connected to rewards or observations)."""

from .certificate import (
    CertificateResult,
    CertificateState,
    CertificateStatus,
    CertificateWitness,
    HalfspaceRegion2D,
    MarginZeroResidual,
    RecoverabilityConfig,
    WitnessResidual,
    certify_recoverability,
    check_witness,
    terminal_contains,
)
from .recovery_manager import (
    RecoveryEpisodeLog,
    RecoveryExitReason,
    RecoveryManager,
    RecoveryProgressKind,
    RecoveryState,
    RecoveryTransition,
    RecoveryUpdate,
)
from .push_curriculum import (
    CurriculumRecoveryOutcome,
    CurriculumUpgradeReason,
    LevelRecoveryStatistics,
    PushCurriculumController,
)
from .stage2_reward import (
    RecoveryEventReward,
    Stage2RecoveryRewardChannel,
    certificate_level,
    certificate_potential,
    certificate_potential_tensor,
    normalized_certificate_margin,
)

__all__ = [
    "CertificateResult",
    "CertificateState",
    "CertificateStatus",
    "CertificateWitness",
    "HalfspaceRegion2D",
    "MarginZeroResidual",
    "RecoverabilityConfig",
    "WitnessResidual",
    "certify_recoverability",
    "check_witness",
    "terminal_contains",
    "RecoveryEpisodeLog",
    "RecoveryExitReason",
    "RecoveryManager",
    "RecoveryProgressKind",
    "RecoveryState",
    "RecoveryTransition",
    "RecoveryUpdate",
    "CurriculumRecoveryOutcome",
    "CurriculumUpgradeReason",
    "LevelRecoveryStatistics",
    "PushCurriculumController",
    "RecoveryEventReward",
    "Stage2RecoveryRewardChannel",
    "certificate_level",
    "certificate_potential",
    "certificate_potential_tensor",
    "normalized_certificate_margin",
]
