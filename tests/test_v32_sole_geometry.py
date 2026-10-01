from pathlib import Path
import sys
import unittest
import numpy as np
from pxr import Usd, UsdGeom
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tasks.manager_based.humanoid_robot_policy_rsl_rl.humanoid_robot_policy_rsl_rl_env_cfg import FOOT_SOLE_VERTICES

class SoleGeometryTests(unittest.TestCase):
    def test_sole_extents_match_v32_collision_mesh_in_foot_frame(self):
        stage=Usd.Stage.Open(str(ROOT/'v3.2/v3.2.usd'))
        cache=UsdGeom.XformCache()
        for index,name in enumerate(['l_ankle_roll_link','r_ankle_roll_link']):
            foot=stage.GetPrimAtPath('/yuerobot/'+name)
            meshes=[p for p in Usd.PrimRange(foot,Usd.TraverseInstanceProxies())
                    if '/collisions/' in str(p.GetPath()) and p.IsA(UsdGeom.Mesh)]
            self.assertEqual(len(meshes),1)
            transform,_=cache.ComputeRelativeTransform(meshes[0],foot)
            pts=np.array([tuple(transform.Transform(v)) for v in UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get()])
            bottom=pts[pts[:,2] < pts[:,2].min()+1e-6]
            configured=np.array(FOOT_SOLE_VERTICES[index])
            np.testing.assert_allclose(configured.min(axis=0),bottom.min(axis=0),atol=1e-6,rtol=0)
            np.testing.assert_allclose(configured.max(axis=0),bottom.max(axis=0),atol=1e-6,rtol=0)

if __name__=='__main__':unittest.main()
