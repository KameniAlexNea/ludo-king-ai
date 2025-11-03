from .config import config
from .player import Player, Piece
from .reward import compute_move_rewards


class MoveManagement:
    def __init__(self, players: list[Player]):
        self.players = players

    def is_safe(self, position: int):
        return position in config.SAFE_SQUARES_ABS

    def get_absolute_position(self, player_index: int, relative_pos: int):
        """
        Converts a player's relative position (1-51) to an absolute board position.
        Returns -1 for yard, home column, or finished.
        """
        if not 1 <= relative_pos <= 51:
            return -1
        start = self.players[player_index].start_square
        # (start + relative_pos - 2) % 52 + 1
        return ((start + relative_pos - 2 + 52) % 52) + 1

    def get_agent_relative_pos(self, agent_index: int, abs_pos: int):
        """
        Converts an absolute board position to a target agent's relative position.
        This is the inverse of get_absolute_position.
        """
        if not 1 <= abs_pos <= 52:
            return -1  # Not on the main track
        agent_start = self.players[agent_index].start_square
        # (abs_pos - agent_start + 52) % 52 + 1
        return ((abs_pos - agent_start + 52) % 52) + 1

    def get_valid_moves(self, player_index: int, dice_roll: int):
        """Finds all valid moves for a player given a dice roll."""
        player = self.players[player_index]
        valid_moves = []

        for piece in player.pieces:
            new_pos = -1

            if piece.position == 0:  # In Yard
                if dice_roll == 6:
                    new_pos = 1  # Start square
            elif piece.position == 57:  # Finished
                continue  # No moves
            elif piece.position > 51:  # On Home Column
                if piece.position + dice_roll <= 57:
                    new_pos = piece.position + dice_roll
            else:  # On Main Track
                if piece.position + dice_roll > 51:
                    # --- FIX: Added missing overflow calculation ---
                    # Entering home column
                    overflow = (piece.position + dice_roll) - 51
                    if overflow <= 6:  # 51 + 6 = 57
                        new_pos = 51 + overflow
                    # --- END FIX ---
                else:
                    new_pos = piece.position + dice_roll

            if new_pos == -1:
                continue

            # --- FIX: Check for self-blockade ---
            # The home square (57) can hold all pieces and is not a blockade.
            if new_pos == 57:
                valid_moves.append(
                    {"piece": piece, "new_pos": new_pos, "dice_roll": dice_roll}
                )
            else:
                # Check for self-blockade (can't move onto your own 2-piece blockade)
                pieces_at_new_pos = sum(
                    1 for p in player.pieces if p.position == new_pos
                )
                if pieces_at_new_pos < 2:
                    valid_moves.append(
                        {"piece": piece, "new_pos": new_pos, "dice_roll": dice_roll}
                    )
            # --- END FIX ---

        return valid_moves

    def make_move(self, player_index: int, piece: Piece, new_position: int, dice_roll: int):
        """Executes a move and returns the reward, events, and extra turn flag."""
        events = {
            "knockouts": [],
            "blockades": [],
            "hit_blockade": False,
            "exited_home": False,
            "finished": False,
            "move_resolved": True,
        }
        player = self.players[player_index]

        # Store old pos for reward calculation and blockade checks
        old_position = piece.position

        if old_position == 0 and new_position == 1:
            events["exited_home"] = True
        if new_position == 57:
            events["finished"] = True

        piece.position = new_position

        hit_blockade = False

        # Check for captures, only on main track (1-51)
        abs_pos = self.get_absolute_position(player_index, new_position)
        if abs_pos != -1 and not self.is_safe(abs_pos):
            for opp_index in range(config.NUM_PLAYERS):
                if opp_index == player_index:
                    continue

                opponent = self.players[opp_index]
                pieces_on_square: list[Piece] = []
                for opp_piece in opponent.pieces:
                    opp_abs_pos = self.get_absolute_position(
                        opp_index, opp_piece.position
                    )
                    if opp_abs_pos == abs_pos:
                        pieces_on_square.append(opp_piece)

                if len(pieces_on_square) == 1:
                    # Capture!
                    pieces_on_square[0].position = 0  # Send to yard
                    events["knockouts"].append(
                        {
                            "player": opp_index,
                            "piece_id": pieces_on_square[0].piece_id,
                            "abs_pos": abs_pos,
                        }
                    )
                elif len(pieces_on_square) == 2:
                    # Hit a blockade!
                    piece.position = old_position  # Move fails, revert position
                    events["hit_blockade"] = True
                    events["move_resolved"] = False
                    hit_blockade = True
                    break  # Move ends

                if hit_blockade:
                    break

        final_position = piece.position

        # Check for forming a blockade at the final position (excluding yard/home)
        if final_position not in (0, 57):
            pieces_at_final_pos = sum(1 for p in player.pieces if p.position == final_position)
            if pieces_at_final_pos == 2:
                events["blockades"].append(
                    {"player": player_index, "relative_pos": final_position}
                )

        rewards = compute_move_rewards(
            len(self.players),
            player_index,
            old_position,
            final_position,
            events,
        )

        mover_reward = rewards[player_index]

        extra_turn = (
            events["finished"]
            or bool(events["knockouts"])
            or dice_roll == 6
        )

        return {
            "reward": mover_reward,
            "rewards": rewards,
            "events": events,
            "extra_turn": extra_turn,
        }
