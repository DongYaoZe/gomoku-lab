import copy
import unittest

from ai_advanced import AdvancedAI
from core import GomokuGame


class VCFRegressionTests(unittest.TestCase):
    def setUp(self):
        self.game = GomokuGame()
        self.ai = AdvancedAI(1)

    def stones(self, player, coords):
        for r, c in coords:
            self.game.simulate_move(r, c, player)

    def test_open_four_has_two_real_winning_completions(self):
        self.stones(1, [(7, 5), (7, 6), (7, 7), (7, 8)])
        self.assertEqual(self.ai._vcf_winning_completions(self.game, 1), [(7, 4), (7, 9)])

    def test_rush_four_has_exactly_one_forced_block(self):
        self.stones(2, [(7, 4)])
        self.stones(1, [(7, 5), (7, 6), (7, 7), (7, 8)])
        self.assertEqual(self.ai._vcf_winning_completions(self.game, 1), [(7, 9)])

    def test_nonforcing_rush_four_does_not_claim_vcf(self):
        self.stones(2, [(7, 4)])
        self.stones(1, [(7, 5), (7, 6), (7, 7)])
        self.assertIsNone(self.ai.find_vcf(self.game, 1, depth=5))

    def test_side_to_move_takes_own_win_before_opponent_threat(self):
        self.stones(1, [(5, 5), (5, 6), (5, 7), (5, 8)])
        self.stones(2, [(9, 5), (9, 6), (9, 7), (9, 8)])
        move = self.ai.find_vcf(self.game, 1, depth=5)
        self.assertIn(move, {(5, 4), (5, 9)})

    def test_black_overline_completion_is_illegal_in_forbidden_mode(self):
        self.game.forbidden_enabled = True
        self.stones(1, [(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
        self.assertFalse(self.ai._vcf_move_is_legal(self.game, 1, 7, 6))

    def test_exact_five_remains_legal_in_forbidden_mode(self):
        self.game.forbidden_enabled = True
        self.stones(1, [(7, 3), (7, 4), (7, 5), (7, 6)])
        self.assertTrue(self.ai._vcf_move_is_legal(self.game, 1, 7, 7))

    def test_search_restores_board_hash_count_and_player(self):
        self.stones(2, [(7, 4)])
        self.stones(1, [(7, 5), (7, 6), (7, 7)])
        before = (copy.deepcopy(self.game.board), self.game.hash, self.game.move_count, self.game.current_player)
        self.ai.find_vcf(self.game, 1, depth=5)
        after = (self.game.board, self.game.hash, self.game.move_count, self.game.current_player)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
