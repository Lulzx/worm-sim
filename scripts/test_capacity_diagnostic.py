import unittest
import numpy as np
from diagnose_capacity_residuals import calibration_scores


class CalibrationTests(unittest.TestCase):
    def test_matches_weighted_least_squares_and_handles_zero_prediction(self):
        rng=np.random.default_rng(3)
        prediction=rng.normal(size=(20,4));prediction[:,3]=0.
        truth=prediction*np.array([2.,-3.,.2,0.])+rng.normal(size=(20,4))*.1
        weight=rng.uniform(.1,1.,size=(20,4))
        signal=np.sum(weight*truth**2,axis=0)
        cross=np.sum(weight*prediction*truth,axis=0)
        power=np.sum(weight*prediction**2,axis=0)
        signed,positive,error,positive_error,signed_error=calibration_scores(signal,cross,power)
        fitted=np.array([np.linalg.lstsq((prediction[:,i]*np.sqrt(weight[:,i]))[:,None],truth[:,i]*np.sqrt(weight[:,i]),rcond=None)[0][0] for i in range(4)])
        np.testing.assert_allclose(signed,fitted,atol=1e-14)
        np.testing.assert_allclose(positive,np.maximum(fitted,0.),atol=1e-14)
        for scale,expected in [(np.ones(4),error),(positive,positive_error),(signed,signed_error)]:
            np.testing.assert_allclose(np.sum(weight*(truth-prediction*scale)**2,axis=0),expected,atol=1e-13)
        self.assertEqual(positive[1],0.)
        self.assertEqual(signed[3],0.)
        self.assertTrue(np.all(signed_error<=positive_error+1e-13))
        self.assertTrue(np.all(positive_error<=error+1e-13))

if __name__=='__main__':unittest.main()
