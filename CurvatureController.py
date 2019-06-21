'''
    Author: Izzat Kamarudzaman
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Code visualises three-tubed concentric tube continuum robot.
'''

import numpy as np


class Jacobian(object):

    def __init__(self, delta_Uz = [], x = [], q = [], Uz = [], model=0):
        self.delta_Uz = delta_Uz
        self.x = x
        self.q = q
        self.model = model
        self.Uz = Uz
        self.J = np.zeros((len(self.Uz), len(self.Uz)), dtype=np.float)

    def dxdq(self, uz):
        # return xx.transpose() * qq
        (r1,r2,r3,Uz) = self.model(self.q, uz)
        # xx = np.array(r1[-1])                               # TODO: check!
        return Uz

    def f(self, uz):
        return np.array(self.dxdq(uz), dtype = np.float)

    def jac_approx(self):
        for i in np.arange(len(self.Uz)):
            uz_iplus = self.Uz.copy()
            uz_iminus = self.Uz.copy()

            uz_iplus[i] += self.delta_Uz[i] / 2
            uz_iminus[i] -= self.delta_Uz[i] / 2

            f_plus = self.f(uz_iplus)
            f_minus = self.f(uz_iminus)

            self.J[:, i] = np.array((f_plus - f_minus) / self.delta_Uz[i]).flatten()

    def p_inv(self):
        self.jac_approx()
        jac_inv = np.linalg.pinv(self.J)  # inv(X^T * X) * X^T
        return jac_inv


class Controller(object):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


if __name__ == "__main__":

    import CTR_model
    model = lambda q, uz_0: CTR_model.moving_CTR(q, uz_0)

    uz_0 = np.array([[0.0, 0.0, 0.0]]).transpose()
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # inputs q
    delta_Uz = np.ones(3) * 1e-3
    x = np.zeros(3)

    u_jac = Jacobian(delta_Uz, x, q, uz_0, model)
    # u_jac.jac_approx()
    J = u_jac.J
    J_inv = u_jac.p_inv()

    print('J:\n', J)
    print('\nJ_inv:\n', J_inv)
    print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(J_inv, J))))
    print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv, np.dot(J_inv, np.dot(J, J_inv))))

    
    # q = J_inv * 