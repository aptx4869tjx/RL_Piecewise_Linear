class HybridAT():
    def __init__(self, sys_data, action_dim) -> None:
        super().__init__()
        self.sys_data = sys_data
        self.modes = None
        self.jumps = None
        self.modes_str = None
        self.jumps_str = None
        self.setting_str = None
        self.init_str = None
        self.hybrid_str = None
        self.action_dim = action_dim

        # self.divide_tool = divide_tool
        # self.network = network

    def generate(self, divide_tool, network):
        return 0

    def output_modes(self):
        if self.modes_str is None:
            self.generate_modes_str()
        print(self.modes_str)

    def generate_modes_str(self):
        modes_str = 'modes\n{\n'
        modes_str += self.modes[0].mode_str + '}\n'
        self.modes_str = modes_str

    def output_jumps(self):
        if self.jumps_str is None:
            self.generate_jumps_str()
        print(self.jumps_str)

    def generate_jumps_str(self):
        jumps_str = 'jumps\n{\n'
        for jump in self.jumps:
            jumps_str += jump.jump_str
        jumps_str += '}\n'
        self.jumps_str = jumps_str

    def generate_setting_str(self):
        setting_str = 'setting\n{\n'
        # setting_str += 'fixed steps 0.02\n'
        # setting_str += 'time 100\n'
        # setting_str += 'remainder estimation 1e-5\n'
        setting_str += 'fixed steps ' + str(self.sys_data['fixed_step']) + '\n' \
                                                                           'time 150\n' \
                                                                           'remainder estimation 1e-6\n' \
                                                                           'identity precondition\n' \
                       + self.sys_data['plot'] + ' octagon ' + self.sys_data['state_vars'][0] + ',' + \
                       self.sys_data['state_vars'][1] + '\n' \
                                                        'adaptive orders { min 3, max 8 }\n' \
                                                        'cutoff 1e-17\n' \
                                                        'precision 100\n' \
                                                        'output out\n' \
                                                        'max jumps ' + str(
            self.sys_data['max_control_depth']) + '\n' \
                                                  'print on\n}\n'
        self.setting_str = setting_str

    def output_setting_str(self):
        if self.setting_str is None:
            self.generate_setting_str()
        print(self.setting_str)

    def generate_init_str(self):
        init_str = 'init\n{\n'
        init_str += self.sys_data['name'] + '\n{\n'
        for i, state_var in enumerate(self.sys_data['state_vars']):
            init_str += state_var + ' in ' + str(self.sys_data['init'][i]) + '\n'
        if self.action_dim == 1:
            init_str += 'a in [0, 0]\n'
        elif self.action_dim > 1:
            for i in range(self.action_dim):
                init_str += 'a' + str(i + 1) + ' in [0, 0]\n'
        cs = str(self.sys_data['control_step'])
        init_str += 'clock in [' + cs + ', ' + cs + ']\n}\n}\n'
        self.init_str = init_str

    def output_init_str(self):
        if self.init_str is None:
            self.generate_init_str()
        print(self.init_str)

    def generate_bybrid_str(self):
        if self.setting_str is None:
            self.generate_setting_str()
        if self.modes_str is None:
            self.generate_modes_str()
        if self.jumps_str is None:
            self.generate_jumps_str()
        if self.init_str is None:
            self.generate_init_str()
        hybrid_str = 'hybrid reachability\n{\n'
        hybrid_str += 'state var '
        for state_var in self.sys_data['state_vars']:
            hybrid_str += state_var + ', '
        if self.action_dim == 1:
            hybrid_str += 'a, '
        elif self.action_dim > 1:
            for i in range(self.action_dim):
                hybrid_str += 'a' + str(i + 1) + ', '
        hybrid_str += 'clock\n'
        hybrid_str += self.setting_str
        hybrid_str += self.modes_str
        hybrid_str += self.jumps_str
        hybrid_str += self.init_str
        hybrid_str += '}\n'
        self.hybrid_str = hybrid_str

    def output_hybrid_str(self):
        if self.hybrid_str is None:
            self.generate_bybrid_str()
        print(self.hybrid_str)
