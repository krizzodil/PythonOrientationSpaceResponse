

class OrientationSpaceResponse:
    def __init__(self, filter, angularResponse):
        self.filter = filter
        self.angularResponse = angularResponse
        self.n = self.angularResponse.shape[2]

    def get_a(self):
        return self.angularResponse
    def set_a(self, a):
        self.angularResponse = a
    a = property(get_a, set_a)

