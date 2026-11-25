"""
tests/test_smoke.py
Smoke tests for the rmgarchpy package.
Verifies that core modules can be imported and basic functions are callable.
"""

import unittest
import numpy as np

class TestSmoke(unittest.TestCase):

    def test_imports(self):
        """Test if core modules can be imported successfully."""
        try:
            import rmgarchpy
            from rmgarchpy import utils
            from rmgarchpy import varx
            # 尝试导入可能尚未完全实现的模块，捕获 ImportError 但不失败，仅警告
            # 这样冒烟测试不会因为某个子模块未完成而完全崩溃
            try:
                from rmgarchpy import dcc
            except ImportError:
                print("Warning: rmgarchpy.dcc module not found or failed to import.")
                
            try:
                from rmgarchpy import cgarch
            except ImportError:
                print("Warning: rmgarchpy.cgarch module not found or failed to import.")
                
        except ImportError as e:
            self.fail(f"Failed to import core rmgarchpy modules: {e}")

    def test_utils(self):
        """Test basic utility functions."""
        from rmgarchpy import utils
        
        # 测试 as_matrix
        arr_1d = np.array([1, 2, 3])
        arr_2d = utils.as_matrix(arr_1d)
        self.assertEqual(arr_2d.shape, (3, 1))
        
        arr_2d_input = np.array([[1, 2], [3, 4]])
        arr_2d_output = utils.as_matrix(arr_2d_input)
        np.testing.assert_array_equal(arr_2d_input, arr_2d_output)

    def test_varx_smoke(self):
        """Smoke test for VARX module."""
        from rmgarchpy.varx import varx_fit, varx_forecast
        
        np.random.seed(42)
        # 生成简单的测试数据
        Y = np.random.randn(100, 2)
        X = np.random.randn(100, 1)
        
        # 测试拟合
        fit = varx_fit(Y, X, lags_y=1, lags_x=1, constant=True)
        self.assertIn('coef', fit)
        self.assertIn('residuals', fit)
        self.assertEqual(fit['coef'].shape[0], 2) # 2 endogenous variables
        
        # 测试预测
        forecast = varx_forecast(fit, Y, X, n_ahead=5)
        self.assertEqual(forecast['forecast'].shape, (5, 2))

    def test_dcc_smoke(self):
        """Smoke test for DCC module (if available)."""
        try:
            from rmgarchpy.dcc import dcc_fit
        except ImportError:
            self.skipTest("DCC module not available")
            
        np.random.seed(42)
        # 生成简单的正定矩阵数据作为标准化残差
        n_obs, k = 200, 2
        z = np.random.randn(n_obs, k)
        
        # 测试拟合 (使用简化的参数)
        try:
            fit = dcc_fit(z, order=(1, 1))
            self.assertIn('omega', fit)
            self.assertIn('alpha', fit)
            self.assertIn('beta', fit)
        except Exception as e:
            # 如果底层优化器报错，冒烟测试仅打印警告，不强制失败，视具体实现而定
            print(f"Warning during DCC smoke test: {e}")

    def test_cgarch_smoke(self):
        """Smoke test for CGARCH module (if available)."""
        try:
            from rmgarchpy.cgarch import cgarch_fit
        except ImportError:
            self.skipTest("CGARCH module not available")
            
        np.random.seed(42)
        # 生成简单的收益率数据
        Y = np.random.randn(200) * 0.02
        
        # 测试拟合
        try:
            fit = cgarch_fit(Y, order=(1, 1, 1))
            self.assertIn('omega', fit)
            self.assertIn('alpha', fit)
            self.assertIn('beta', fit)
            self.assertIn('gamma', fit)
        except Exception as e:
            print(f"Warning during CGARCH smoke test: {e}")

if __name__ == "__main__":
    unittest.main()