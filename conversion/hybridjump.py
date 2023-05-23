class HybridJump():
    # all self loop
    def __init__(self, sys_data) -> None:
        self.guard = None
        self.reset = None
        self.source = sys_data['name']
        self.target = sys_data['name']
        self.jump_str = None

    def output_jump_str(self):
        if self.jump_str is None:
            self.generate_jump_str()
        print(self.jump_str)

    def generate_jump_str(self):
        jump_str = self.source + ' -> ' + self.target + '\n'
        jump_str += 'guard\n{\n'
        jump_str += self.guard
        jump_str += '}\nreset\n{\n'
        jump_str += self.reset
        jump_str += '\n}\n'
        jump_str += 'interval aggregation\n'
        self.jump_str = jump_str
