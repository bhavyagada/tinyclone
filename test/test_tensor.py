import torch
import numpy as np
import unittest
from tinyclone.tensor import Tensor, Conv2d

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinyclone(unittest.TestCase):
  def test_backward_pass(self):
    def test_tinyclone():
      x = Tensor(x_init)
      W = Tensor(W_init)
      m = Tensor(m_init)
      out = x.dot(W).relu()
      out = out.logsoftmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.data, x.grad, W.grad

    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.detach().numpy(), x.grad, W.grad

    for x, y in zip(test_tinyclone(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)
  
  def test_conv2d(self):
    x = torch.randn((1,2,10,7), requires_grad=True)
    w = torch.randn((4,2,3,3), requires_grad=True)
    xt = Tensor(x.detach().numpy())
    wt = Tensor(w.detach().numpy())

    out = torch.nn.functional.conv2d(x,w)
    ret = Conv2d.apply(Conv2d, xt, wt)
    np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)

    out.mean().backward()
    ret.mean().backward()

    np.testing.assert_allclose(w.grad, wt.grad, atol=1e-5)
    np.testing.assert_allclose(x.grad, xt.grad, atol=1e-5)

if __name__ == "__main__":
  unittest.main()
