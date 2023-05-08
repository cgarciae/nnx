from tkinter import Variable


@dataclass
class Linear:
    din: int
    dout: int

    def create_state(self, key) -> State:
        return State(
            kernel=Variable("params", jax.random.uniform(key, (self.din, self.dout))),
            bias=Variable("params", jax.numpy.zeros((self.dout,))),
        )

    def __call__(self, state, x):
        return x @ state.kernel + state.bias


class MLP:
    def __init__(self, din: int, dmid: int, dout: int):
        self.linear1 = Linear(din, dmid)
        self.linear2 = Linear(dmid, dout)

    def create_state(self, rngs) -> State:
        return State(
            linear1=self.linear1.create_state(rngs),
            linear2=self.linear2.create_state(rngs),
        )

    def __call__(self, state, rngs, x):
        x = nn.relu(self.linear1(state.linear1, x))
        x = self.linear2(state.linear2, x)

        state.output = Variable("i", x)
        return x
