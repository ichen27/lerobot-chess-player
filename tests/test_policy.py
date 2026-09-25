from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from chess_perception.game import ChessGame

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

class Policy(torch.nn.Module):
    def __init__(self, output=None):
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
        self.output = torch.arange(6, dtype=torch.float32).unsqueeze(0) if output is None else output
        self.seen = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, observation):
        self.seen.append(observation)
        return self.output

def make_game(policy, state=None):
    game = ChessGame("unused.pt")
    game.pi0_policy = policy
    game.policy_steps = 2
    game.policy_fps = 30
    sent = []
    game.robot = SimpleNamespace(get_observation=lambda: dict.fromkeys(JOINTS, 5.0) if state is None else state,
                                 send_action=sent.append)
    game.capture_image = lambda: np.array([[[10,20,30]]], dtype=np.uint8)  # OpenCV BGR
    return game, sent

def test_policy_uses_repeated_observations_rgb_and_batched_language(monkeypatch):
    policy = Policy()
    game, sent = make_game(policy)
    monkeypatch.setattr("chess_perception.game.time.sleep", lambda _: None)
    game._pi0_execute("pick black pawn from e7, place on e5")
    assert len(sent) == 2
    assert sent[0] == dict(zip(JOINTS, range(6)))
    assert policy.resets == 1
    assert len(policy.seen) == 2
    assert policy.seen[0]["task"] == ["pick black pawn from e7, place on e5"]
    assert torch.allclose(policy.seen[0]["observation.images.innomaker"][0,:,0,0],
                          torch.tensor([30,20,10]) / 255)

@pytest.mark.parametrize("output", [torch.full((1,6), float("nan")), torch.zeros(1,5), torch.zeros(2,6)])
def test_invalid_policy_output_never_reaches_arm(output):
    game, sent = make_game(Policy(output))
    with pytest.raises(RuntimeError):
        game._pi0_execute("move")
    assert sent == []

def test_missing_joint_feedback_never_becomes_zero_joint_command():
    game, sent = make_game(Policy(), state={})
    with pytest.raises(RuntimeError):
        game._pi0_execute("move")
    assert sent == []
