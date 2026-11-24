import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'closed_loop'))
import numpy as np
from core import execute_prefix,pose_matrix
class ExecutionTests(unittest.TestCase):
    def test_rotated_translation(self):
        result=execute_prefix([10,20,1,math.pi/2],[[2,0,0]]*8,np.arange(1,9)*.5)
        np.testing.assert_allclose(result[-1],[10,22,1,math.pi/2])
        np.testing.assert_allclose(result[0,:2],[10,20.4])
    def test_feedback_not_reset(self):
        a=execute_prefix([0,0,0,0],[[1,0,0]]*8,np.arange(1,9)*.5)
        b=execute_prefix(a[-1],[[1,0,0]]*8,np.arange(1,9)*.5)
        self.assertEqual(b[-1,0],2)
    def test_bad_output_rejected(self):
        with self.assertRaises(ValueError):execute_prefix([0,0,0,0],[[float('nan'),0,0]]*8,np.arange(1,9)*.5)
    def test_future_access_rejected(self):
        from worker import deny_future
        with self.assertRaises(RuntimeError):deny_future('open',('/data/scene/future/rgb/CAM_FRONT/000010.jpg',))
        with self.assertRaises(RuntimeError):deny_future('open',('/data/scene/future/ego_trajectory.csv',))
        deny_future('open',('/data/scene/history/ego.csv',))
    def test_heading_shortest_arc(self):
        result=execute_prefix([0,0,0,0],[[0,0,2*math.pi-.1]]*8,np.arange(1,9)*.5)
        self.assertAlmostEqual(result[-1,3],-.1)
if __name__=='__main__':unittest.main()
