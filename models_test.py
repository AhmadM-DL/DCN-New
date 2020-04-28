from models import *
import unittest
from deep_learning_unittest import *


class ModelsTest(unittest.TestCase):

    def test_dynamic_da(self):
        da = dA(n_visible=28*28, n_hidden=30, corruption_level=0.1)

        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, da.parameters()),
            lr=0.01,
        )

        dummy_input = torch.rand((10, 28*28))
        batch = [dummy_input, dummy_input]

        loss_fn = torch.nn.MSELoss()

        device = torch.device("cpu")

        do_forward_step(da, batch=batch, device=device)

        do_train_step(da, loss_fn=loss_fn, optim=optimizer,
                      batch=batch, device=device)

        test_param_change(vars_change=True, model=da, loss_fn=loss_fn,
                          optim=optimizer, batch=batch, device=device)

        test_param_tied(param1=da.encoder[0].weight,
                        param2=da.decoder[0].weight.transpose(0, 1),
                        model = da,
                        loss_fn=loss_fn,
                        optim=optimizer,
                        batch=batch,
                        device=device)

        assert True


if __name__ == "main":
    unittest.main()
