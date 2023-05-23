class HybridMode():
    def __init__(self, name) -> None:
        self.name = name
        self.dynamics = None
        self.invariant = None
        self.form = 'nonpoly ode'
        self.mode_str = None

    def output_mode_str(self):
        if self.mode_str is None:
            self.genarate_mode_str()
        print(self.mode_str)

    def generate_mode_str(self):
        assert self.dynamics is not None
        assert self.invariant is not None
        mode_str = self.name + '\n{\n'
        mode_str += self.form + '\n{\n'
        mode_str += self.dynamics
        mode_str += '\n}\ninv\n{\n'
        mode_str += self.invariant
        mode_str += '\n}\n}\n'
        self.mode_str = mode_str



