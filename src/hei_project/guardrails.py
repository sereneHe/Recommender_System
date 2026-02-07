import torch
import wandb
import logging


# Cloud configs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gcp_logger")


class DataGuard:
    def __init__(self, strictness: float = 2.0):
        # We expect data between 0 and 1.
        # strictness=2.0 allows a little bit of noise (up to 3.0 or -2.0),
        # but blocks wild outliers.
        self.min_allowed = 0.0 - strictness
        self.max_allowed = 1.0 + strictness

    def validate(self, x: torch.Tensor) -> bool:
        # Check for NaNs (Empty values)
        if torch.isnan(x).any():
            self._trigger_alert("Input contains NaNs! (Empty values detected")
            return False

        # Check for Infinite values
        if torch.isinf(x).any():
            self._trigger_alert("Input contains Infinite values")
            return False

        # Check for massive outliers (Data Drift)
        # If the model expects 0.5 and gets 500, it will output garbage.
        if x.max() > self.max_allowed or x.min() < self.min_allowed:
            msg = (
                f"Data Drift Detected! Input is wildly out of distribution.\n"
                f"Expected range: [0, 1]\n"
                f"Received range: [{x.min():.2f}, {x.max():.2f}]\n"
                f"Process aborted to prevent model hallucination."
            )
            self._trigger_alert(msg)
            return False

        return True

    def _trigger_alert(self, message: str):
        """Logs the error and sends a W&B Alert if active."""
        logger.warning(message)

        # This sends an email/slack notification if WandB is running
        if wandb.run is not None:
            wandb.alert(  # type: ignore
                title="Bad Input Detected",
                text=message,
                level=wandb.AlertLevel.WARN,  # type: ignore
                wait_duration=300,  # Wait 5 mins before sending another alert
            )
