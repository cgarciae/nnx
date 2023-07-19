import copy

from nnx import ids


class TestIds:

  def test_hashable(self):
    id1 = ids.uuid()
    id2 = ids.uuid()
    assert id1 == id1
    assert id1 != id2
    assert hash(id1) != hash(id2)
    id1c = copy.copy(id1)
    id1dc = copy.deepcopy(id1)
    assert hash(id1) != hash(id1c)
    assert hash(id1) != hash(id1dc)
