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
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from lerobot.policies.factory import make_policy
            from lerobot.configs.policies import PreTrainedConfig

            config = PreTrainedConfig.from_pretrained(model_path)
            config.pretrained_path = model_path
            self.pi0_policy = PI0Policy.from_pretrained(model_path)
            self.pi0_policy.eval()
            print("[setup] Pi0 policy loaded successfully.")
        except Exception as e:
            print(f"[setup] WARNING: Failed to load Pi0 policy: {e}")
            print("[setup] Falling back to print-only mode.")
            self.pi0_policy = None
            return

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
            print(f"[setup] WARNING: Failed to connect robot: {e}")
            print("[setup] Pi0 will run but cannot send actions to arm.")
            self.robot = None

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

            result = self.pipeline.process_frame(image, color="white")

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

            # Generate move description(s) for the robot arm.
            move_commands = self._build_move_commands(robot_chess_move)

            for cmd in move_commands:
                print(f"[robot] {cmd}")

            # Execute via Pi0 or print-only.
            for cmd in move_commands:
                self.execute_robot_move(cmd)

            # Apply the move to internal state.
            self.board.push(robot_chess_move)
            self.prev_fen = self.board.fen()

            print(f"[robot] Played: {best_move}")
            print(self.board)
            print()

            # Optionally verify the board after robot moves.
            if self.pi0_policy is not None:
                print("[verify] Waiting for arm to settle ...")
                time.sleep(2.0)
                try:
                    verify_image = self.capture_image()
                    verify_result = self.pipeline.process_frame(
                        verify_image, color="white"
                    )
                    if verify_result["fen"]:
                        print(f"[verify] Post-move FEN: {verify_result['fen']}")
                except RuntimeError as e:
                    print(f"[verify] Warning: {e}")

            move_number += 1

        # Game over.
        self._announce_result()
        self._cleanup()

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
        prev_board = chess.Board(prev_fen)
        new_board = chess.Board(new_fen)

        prev_pieces: dict[int, Optional[chess.Piece]] = {}
        new_pieces: dict[int, Optional[chess.Piece]] = {}

        for sq in chess.SQUARES:
            prev_pieces[sq] = prev_board.piece_at(sq)
            new_pieces[sq] = new_board.piece_at(sq)

        # Find squares that changed.
        emptied = []   # had a piece before, empty now
        filled = []    # empty before, has a piece now
        changed = []   # had a piece before, different piece now

        for sq in chess.SQUARES:
            old_p = prev_pieces[sq]
            new_p = new_pieces[sq]
            if old_p is not None and new_p is None:
                emptied.append(sq)
            elif old_p is None and new_p is not None:
                filled.append(sq)
            elif old_p is not None and new_p is not None and old_p != new_p:
                changed.append(sq)

        # Try to match the diff against legal moves.
        # This is the most reliable approach: enumerate legal moves and see
        # which one produces the observed board change.
        for move in prev_board.legal_moves:
            test_board = prev_board.copy()
            test_board.push(move)
            # Compare piece placement (ignore castling rights, en passant, etc.)
            match = True
            for sq in chess.SQUARES:
                if test_board.piece_at(sq) != new_board.piece_at(sq):
                    match = False
                    break
            if match:
                return move.uci()

        # Fallback heuristic: simple one-piece move.
        if len(emptied) == 1 and len(filled) == 1 and not changed:
            return chess.square_name(emptied[0]) + chess.square_name(filled[0])

        # Fallback: one emptied, one changed (capture).
        if len(emptied) == 1 and not filled and len(changed) == 1:
            return chess.square_name(emptied[0]) + chess.square_name(changed[0])

        # Castling: two emptied, two filled.
        if len(emptied) == 2 and len(filled) == 2:
            # Find the king among emptied squares.
            for sq in emptied:
                piece = prev_pieces[sq]
                if piece and piece.piece_type == chess.KING:
                    # King's destination is whichever filled square is on the
                    # same rank.
                    king_rank = chess.square_rank(sq)
                    for dst in filled:
                        if chess.square_rank(dst) == king_rank:
                            file_diff = chess.square_file(dst) - chess.square_file(sq)
                            if abs(file_diff) == 2:
                                return chess.square_name(sq) + chess.square_name(dst)
            # Generic fallback for castling.
            return chess.square_name(emptied[0]) + chess.square_name(filled[0])

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
        import torch

        print(f"[pi0] Executing: {instruction}")
        try:
            # Get current robot observation (joint positions + camera)
            obs = self.robot.get_observation() if self.robot else {}

            # Build the observation dict Pi0 expects
            image = self.capture_image()
            img_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float() / 255.0
            )

            # Build state vector from robot joint positions
            state_keys = [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
            ]
            state = torch.tensor(
                [obs.get(k, 0.0) for k in state_keys],
                dtype=torch.float32,
            ).unsqueeze(0)

            observation = {
                "observation.images.innomaker": img_tensor,
                "observation.state": state,
                "task": instruction,
            }

            # Run policy inference to get action chunk
            with torch.no_grad():
                action = self.pi0_policy.select_action(observation)

            # action is a dict of {motor.pos: value} or a tensor of shape [chunk, 6]
            # Execute the action chunk step by step
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
                fps = 30
                for step_idx in range(len(action_np)):
                    step_action = {
                        f"{k}.pos": float(action_np[step_idx, i])
                        for i, k in enumerate([
                            "shoulder_pan", "shoulder_lift", "elbow_flex",
                            "wrist_flex", "wrist_roll", "gripper",
                        ])
                    }
                    if self.robot:
                        self.robot.send_action(step_action)
                    time.sleep(1.0 / fps)
            elif isinstance(action, dict):
                if self.robot:
                    self.robot.send_action(action)
                time.sleep(1.0)

            print(f"[pi0] Completed: {instruction}")

        except Exception as e:
            print(f"[pi0] Error during execution: {e}")
            import traceback
            traceback.print_exc()

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
