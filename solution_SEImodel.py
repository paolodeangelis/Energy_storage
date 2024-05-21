class SimpleSEI(pybamm.BaseModel):
    def __init__(self):
        super().__init__()
        self.set_variables()
        self.set_governing_eqs()
        self.set_boudary_conditions()
        self.set_initial_conditions()

    def set_variables(self):
        xi = pybamm.SpatialVariable("xi", domain="SEI layer", coord_sys="cartesian")
        c = pybamm.Variable("Solvent concentration [mol.m-3]", domain="SEI layer")
        L = pybamm.Variable("SEI thickness [m]")
        self.variables.update(
            {
                "xi": xi,
                "Solvent concentration [mol.m-3]": c,
                "SEI thickness [m]": L,
            }
        )

    def get_parameters(self):
        k = pybamm.Parameter("Reaction rate constant [m.s-1]")
        L_0 = pybamm.Parameter("Initial thickness [m]")
        V_hat = pybamm.Parameter("Partial molar volume [m3.mol-1]")
        c_inf = pybamm.Parameter("Bulk electrolyte solvent concentration [mol.m-3]")
        return k, L_0, V_hat, c_inf

    def D(self, c):
        return pybamm.FunctionParameter(
            "Diffusivity [m2.s-1]", {"Solvent concentration [mol.m-3]": c}
        )

    def eps(self, xi):
        return pybamm.FunctionParameter("Porosity", {"SEI Porosity": xi})

    def tau(self, xi):
        return pybamm.FunctionParameter("Tortuosity", {"SEI Tortuosity": xi})

    def set_governing_eqs(self):
        k, L_0, V_hat, c_inf = self.get_parameters()
        xi = self.variables["xi"]
        c = self.variables["Solvent concentration [mol.m-3]"]
        L = self.variables["SEI thickness [m]"]

        # SEI reaction flux
        R = k * pybamm.BoundaryValue(c, "left")

        # solvent concentration equation
        q = -1 / L * self.eps(xi) * self.D(c) / self.tau(xi) * pybamm.grad(c)  # flux
        dcdt = (V_hat * R) / L * pybamm.inner(xi, pybamm.grad(c)) - 1 / L * pybamm.div(
            q
        )

        # SEI thickness equation
        dLdt = V_hat * R
        self.variables.update(
            {
                "SEI growth rate [m]": dLdt,
                "Surface reaction rate concentration [mol.m-2.s-1]": R,
            }
        )

        self.rhs = {L: dLdt, c: dcdt}

    def set_boudary_conditions(self):
        k, L_0, V_hat, c_inf = self.get_parameters()
        c = self.variables["Solvent concentration [mol.m-3]"]
        L = self.variables["SEI thickness [m]"]
        xi = self.variables["xi"]
        D_left = pybamm.BoundaryValue(self.D(c), "left")
        eps_left = pybamm.BoundaryValue(self.eps(xi), "left")
        tau_left = pybamm.BoundaryValue(self.tau(xi), "left")
        # SEI reaction flux
        R = k * pybamm.BoundaryValue(c, "left")
        # Neumann
        grad_c_left = R * L * tau_left / D_left / eps_left
        # Dirichlet
        c_right = c_inf
        self.boundary_conditions = {
            c: {"left": (grad_c_left, "Neumann"), "right": (c_right, "Dirichlet")}
        }

    def set_initial_conditions(self):
        k, L_0, V_hat, c_inf = self.get_parameters()
        c = self.variables["Solvent concentration [mol.m-3]"]
        L = self.variables["SEI thickness [m]"]
        c_init = c_inf
        L_init = L_0
        self.initial_conditions = {c: c_init, L: L_init}
