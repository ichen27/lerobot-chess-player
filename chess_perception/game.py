"""Chess game orchestrator: perception + Stockfish + Pi0 robot arm control."""

from __future__ import annotations

import time
from typing import Optional

import chess
import cv2
import numpy as np

from chess_perception.pipeline import ChessPerceptionPipeline


# FEN piece symbol -> human-readable name mapping.
_FEN_TO_NAME = {
    "P": "white pawn", "N": "white knight", "B": "white bishop",
    "R": "white rook", "Q": "white queen", "K": "white king",
    "p": "black pawn", "n": "black knight", "b": "black bishop",
    "r": "black rook", "q": "black queen", "k": "black king",
}


class ChessGame:
    """Orchestrates a Robot (black) vs Human (white) chess game.

    Ties together the vision-based perception pipeline, Stockfish engine,
    and (optionally) a Pi0 robot arm policy for physical move execution.
    """

    def __init__(
        self,
        yolo_weights: str,
        stockfish_path: str = "stockfish",
        pi0_model_path: Optional[str] = None,
        camera_index: int = 0,
        board_orientation: str = "white_bottom",
        robot_port: str = "/dev/tty.usbmodem11401",
    ):
        self.yolo_weights = yolo_weights
        self.stockfish_path = stockfish_path
        self.pi0_model_path = pi0_model_path
        self.camera_index = camera_index
        self.board_orientation = board_orientation
        self.robot_port = robot_port
        self.policy_steps = 300
        self.policy_fps = 30

        # Initialised in setup().
        self.pipeline: Optional[ChessPerceptionPipeline] = None
        self.pi0_policy = None
        self.robot = None
        self.board: Optional[chess.Board] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_fen: Optional[str] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all components: perception pipeline, camera, Pi0."""
        print("[setup] Initialising chess perception pipeline ...")
        self.pipeline = ChessPerceptionPipeline(
            yolo_weights=self.yolo_weights,
            stockfish_path=self.stockfish_path,
            board_orientation=self.board_orientation,
        )

        print("[setup] Opening camera {} ...".format(self.camera_index))
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[setup] Initialising python-chess board ...")
        self.board = chess.Board()  # standard starting position

        if self.pi0_model_path is not None:
            self._load_pi0_policy(self.pi0_model_path)
        else:
            print("[setup] No Pi0 model path provided -- moves will be printed only.")

        # Capture the initial board state as prev_fen.
        self.prev_fen = self.board.fen()
        print("[setup] Ready. Human plays WHITE, robot plays BLACK.")

    def _load_pi0_policy(self, model_path: str) -> None:
        """Load Pi0 policy and connect to the OMX follower robot arm."""
        print(f"[setup] Loading Pi0 policy from {model_path} ...")
        try:
            from importlib.metadata import version
            if version("lerobot") != "0.3.4":
                raise RuntimeError("This experimental adapter targets LeRobot 0.3.4; see docs/hardware.md.")
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            import torch
            self.pi0_policy = PI0Policy.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pi0_policy.to(device).eval()
            print("[setup] Pi0 policy loaded successfully.")
        except Exception as e:
            self.pi0_policy = None
            raise RuntimeError(f"Failed to load experimental Pi0 policy: {e}") from e

        # Connect to the OMX follower robot
        print(f"[setup] Connecting to OMX follower on {self.robot_port} ...")
        try:
            from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
            from lerobot.robots.omx_follower.omx_follower import OmxFollower

            robot_config = OmxFollowerConfig(
                port=self.robot_port,
                cameras={},  # We use our own camera via OpenCV
            )
            self.robot = OmxFollower(robot_config)
            self.robot.connect()
            self.robot.bus.enable_torque()
            print("[setup] Robot arm connected and torque enabled.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect OMX arm: {e}") from e

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def capture_image(self) -> np.ndarray:
        """Grab a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not open. Call setup() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    # ------------------------------------------------------------------
    # Main game loop
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Run the main game loop until the game ends."""
        if self.board is None:
            self.setup()

        print("\n=== Chess Game Started ===")
        print("Human (WHITE) moves first. Press Enter after each move.\n")
        print(self.board)
        print()

        move_number = 1

        while not self.board.is_game_over():
            # --- Human turn (WHITE) ---
            input(f"Move {move_number}: Make your move on the board, then press Enter ...")

            try:
                image = self.capture_image()
            except RuntimeError as e:
                print(f"[error] {e}")
                continue

            result = self.pipeline.process_frame(image, color="black", analyze=False)

            if result["error"]:
                print(f"[perception] Error: {result['error']}")
                print("[perception] Retrying -- press Enter to re-scan ...")
                continue

            new_fen = result["fen"]
            if new_fen is None:
                print("[perception] Could not determine board state. Try again.")
                continue

            print(f"[perception] Detected FEN: {new_fen}")

            human_move = self.detect_human_move(self.prev_fen, new_fen)
            if human_move is None:
                print("[detect] Could not determine human move from board diff.")
                print("[detect] Please re-position pieces and press Enter to retry.")
                continue

            # Validate the move.
            try:
                chess_move = chess.Move.from_uci(human_move)
            except ValueError:
                print(f"[detect] Invalid UCI move: {human_move}")
                continue

            if chess_move not in self.board.legal_moves:
                print(f"[validate] Move {human_move} is not legal in current position.")
                print("[validate] Legal moves:", " ".join(
                    m.uci() for m in self.board.legal_moves
                ))
                continue

            # Apply human move.
            self.board.push(chess_move)
            print(f"[human] Played: {human_move}")
            print(self.board)
            print()

            if self.board.is_game_over():
                break

            # --- Robot turn (BLACK) ---
            print("[engine] Thinking ...")
            best_move = self.pipeline.engine.get_best_move(
                self.board.fen(), color="black"
            )

            if best_move is None:
                print("[engine] No legal moves available for black.")
                break

            robot_chess_move = chess.Move.from_uci(best_move)

            self.execute_robot_turn(robot_chess_move)
            print(f"[robot] Played: {best_move}")
            print(self.board)
            print()

            move_number += 1

        # Game over.
        self._announce_result()

    # ------------------------------------------------------------------
    # Move detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_human_move(prev_fen: str, new_fen: str) -> Optional[str]:
        """Compare two FEN strings and return the move in UCI notation.

        Handles normal moves, captures, castling, and en passant.

        Args:
            prev_fen: FEN string before the human's move.
            new_fen: FEN string after the human's move (from perception).

        Returns:
            UCI move string (e.g. "e2e4") or None if detection fails.
        """
        try:
            prev_board = chess.Board(prev_fen)
            new_board = chess.Board(new_fen)
        except ValueError:
            return None
        if not prev_board.is_valid():
            return None
        # Only placement is observed. Preserve history in the authoritative board.
        for move in prev_board.legal_moves:
            candidate = prev_board.copy()
            candidate.push(move)
            if candidate.board_fen() == new_board.board_fen():
                return move.uci()
        return None

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def _build_move_commands(self, move: chess.Move) -> list[str]:
        """Build robot arm command strings for a move.

        Handles captures by generating a removal command first.

        Args:
            move: The chess.Move to execute.

        Returns:
            List of command strings for the robot arm.
        """
        commands = []
        fen = self.board.fen()
        board = chess.Board(fen)

        source_sq = move.from_square
        target_sq = move.to_square
        piece = board.piece_at(source_sq)

        if piece is None:
            return [f"pick piece from {chess.square_name(source_sq)}, "
                    f"place on {chess.square_name(target_sq)}"]

        piece_name = _FEN_TO_NAME.get(piece.symbol(), "piece")

        # Handle captures.
        captured = board.piece_at(target_sq)

        # En passant: the captured pawn is not on the target square.
        if (piece.piece_type == chess.PAWN
                and chess.square_file(source_sq) != chess.square_file(target_sq)
                and captured is None):
            # En passant capture.
            ep_sq = target_sq + (-8 if piece.color == chess.WHITE else 8)
            ep_piece = board.piece_at(ep_sq)
            if ep_piece:
                ep_name = _FEN_TO_NAME.get(ep_piece.symbol(), "piece")
                commands.append(
                    f"remove {ep_name} from {chess.square_name(ep_sq)}"
                )
        elif captured is not None:
            cap_name = _FEN_TO_NAME.get(captured.symbol(), "piece")
            commands.append(
                f"remove {cap_name} from {chess.square_name(target_sq)}"
            )

        commands.append(
            f"pick {piece_name} from {chess.square_name(source_sq)}, "
            f"place on {chess.square_name(target_sq)}"
        )

        # Castling: also move the rook.
        if piece.piece_type == chess.KING and abs(
            chess.square_file(source_sq) - chess.square_file(target_sq)
        ) == 2:
            if chess.square_file(target_sq) > chess.square_file(source_sq):
                # Kingside.
                rook_from = chess.square(7, chess.square_rank(source_sq))
                rook_to = chess.square(5, chess.square_rank(source_sq))
            else:
                # Queenside.
                rook_from = chess.square(0, chess.square_rank(source_sq))
                rook_to = chess.square(3, chess.square_rank(source_sq))

            rook = board.piece_at(rook_from)
            rook_name = _FEN_TO_NAME.get(rook.symbol(), "rook") if rook else "rook"
            commands.append(
                f"pick {rook_name} from {chess.square_name(rook_from)}, "
                f"place on {chess.square_name(rook_to)}"
            )

        # Promotion.
        if move.promotion:
            promo_piece = chess.Piece(move.promotion, piece.color)
            promo_name = _FEN_TO_NAME.get(promo_piece.symbol(), "piece")
            commands.append(
                f"promote to {promo_name} on {chess.square_name(target_sq)}"
            )

        return commands

    def handle_capture(self, move: chess.Move, fen: str) -> list[str]:
        """Generate commands for a capturing move.

        Args:
            move: The capturing chess.Move.
            fen: Current board FEN before the move.

        Returns:
            List of command strings: removal first, then pick-and-place.
        """
        board = chess.Board(fen)
        source_sq = move.from_square
        target_sq = move.to_square
        piece = board.piece_at(source_sq)
        captured = board.piece_at(target_sq)

        commands = []

        if captured:
            cap_name = _FEN_TO_NAME.get(captured.symbol(), "piece")
            commands.append(
                f"remove {cap_name} from {chess.square_name(target_sq)}"
            )

        piece_name = _FEN_TO_NAME.get(piece.symbol(), "piece") if piece else "piece"
        commands.append(
            f"pick {piece_name} from {chess.square_name(source_sq)}, "
            f"place on {chess.square_name(target_sq)}"
        )

        return commands

    def execute_robot_turn(self, move: chess.Move) -> None:
        """Commit a move only after manual confirmation or matching perception.

        A failure stops the game. A partially moved physical board requires
        operator recovery; blindly retrying a capture could remove another piece.
        """
        if self.board is None or move not in self.board.legal_moves:
            raise ValueError("Robot move must be legal in the current position.")
        if self.pi0_policy is not None and move.promotion:
            raise RuntimeError("Physical promotion requires manual piece replacement.")
        expected = self.board.copy()
        expected.push(move)
        for command in self._build_move_commands(move):
            self.execute_robot_move(command)

        if self.pi0_policy is None:
            answer = input("Perform the printed move(s), then type y to confirm: ")
            if answer.strip().lower() != "y":
                raise RuntimeError("Move not confirmed; board state was not advanced.")
        else:
            time.sleep(2.0)
            result = self.pipeline.process_frame(self.capture_image(), analyze=False)
            if result["error"] or not result["fen"]:
                raise RuntimeError(f"Post-move verification failed: {result['error']}")
            if chess.Board(result["fen"]).board_fen() != expected.board_fen():
                raise RuntimeError("Post-move verification mismatch; inspect the board before restarting.")
        self.board.push(move)
        self.prev_fen = self.board.fen()

    def execute_robot_move(self, move_description: str) -> None:
        """Send a move command to the Pi0 policy or print it.

        Args:
            move_description: Natural language instruction for the arm,
                e.g. "pick black knight from g8, place on f6".
        """
        if self.pi0_policy is not None:
            self._pi0_execute(move_description)
        else:
            print(f"[execute] (manual) {move_description}")

    def _pi0_execute(self, instruction: str) -> None:
        """Execute a single instruction via Pi0 policy inference.

        Captures camera image + robot state, pairs with language instruction,
        runs Pi0 to generate action chunks, and sends them to the OMX arm.
        """
        if self.robot is None or self.pi0_policy is None:
            raise RuntimeError("A connected arm and loaded policy are required.")
        import torch

        keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ]
        device = next(self.pi0_policy.parameters()).device
        self.pi0_policy.reset()  # Discard queued actions from the previous instruction.
        try:
            for _ in range(self.policy_steps):
                started = time.monotonic()
                obs = self.robot.get_observation()
                state_values = np.asarray([obs[k] for k in keys], dtype=np.float32)
                if not np.isfinite(state_values).all():
                    raise ValueError("Non-finite joint feedback")
                # OpenCV is BGR; Pi0 was trained on RGB images.
                image = cv2.cvtColor(self.capture_image(), cv2.COLOR_BGR2RGB)
                observation = {
                    "observation.images.innomaker": torch.from_numpy(image).permute(
                        2, 0, 1).unsqueeze(0).float().div(255).to(device),
                    "observation.state": torch.from_numpy(state_values).unsqueeze(0).to(device),
                    "task": [instruction],
                }
                with torch.inference_mode():
                    action = self.pi0_policy.select_action(observation)
                # LeRobot 0.3.4 select_action returns ONE action per batch, not a chunk.
                if not isinstance(action, torch.Tensor) or tuple(action.shape) != (1, 6):
                    raise ValueError("Expected a single policy action with shape (1, 6)")
                values = action.detach().float().cpu().numpy()[0]
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite policy action")
                self.robot.send_action(dict(zip(keys, map(float, values))))
                time.sleep(max(0, 1 / self.policy_fps - (time.monotonic() - started)))
        except Exception as exc:
            raise RuntimeError(f"Robot execution failed: {instruction}") from exc
        # A fixed action horizon is not proof of task completion.
        # execute_robot_turn verifies the resulting board before committing.

    # ------------------------------------------------------------------
    # Game end
    # ------------------------------------------------------------------

    def _announce_result(self) -> None:
        """Print the game result."""
        print("\n=== Game Over ===")
        print(self.board)
        print()

        result = self.board.result()
        if self.board.is_checkmate():
            # board.turn is the side that is IN checkmate (it is their turn
            # but they have no legal moves).
            if self.board.turn == chess.WHITE:
                winner = "BLACK (Robot)"
            else:
                winner = "WHITE (Human)"
            print(f"Checkmate! {winner} wins.")
        elif self.board.is_stalemate():
            print("Stalemate. Draw.")
        elif self.board.is_insufficient_material():
            print("Insufficient material. Draw.")
        elif self.board.is_fifty_moves():
            print("Fifty-move rule. Draw.")
        elif self.board.is_repetition():
            print("Threefold repetition. Draw.")
        else:
            print(f"Result: {result}")

    def _cleanup(self) -> None:
        """Release resources."""
        if self.robot is not None:
            try:
                self.robot.disconnect()
                print("[cleanup] Robot arm disconnected.")
            except Exception:
                pass
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        print("[cleanup] Camera released. Game ended.")
