'''
    Author: Izzat Kamarudzaman
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Code visualises three-tubed concentric tube continuum robot.
'''

import numpy as np


class Jacobian(object):

    def __init__(self, delta_q = [], x = [], q = [], uz_0 = [], model=0):
        self.delta_q = delta_q
        self.x = x
        self.q = q
        self.model = model
        self.uz_0 = uz_0
        self.J = np.zeros([len(self.x), len(self.q)], dtype=np.float)

    def dxdq(self, qq):
        # return xx.transpose() * qq
        (r1,r2,r3,Uz) = self.model(qq, self.uz_0)
        xx = np.array(r1[-1])                           # TODO: check!
        return xx

    def f(self, qq):
        return np.array([self.dxdq(qq)], dtype = np.float)

    def jac_approx(self):
        for i in np.arange(len(self.q)):
            q_iplus = self.q.copy()
            q_iminus = self.q.copy()

            q_iplus[i] += self.delta_q[i] / 2
            q_iminus[i] -= self.delta_q[i] / 2

            f_plus = self.f(q_iplus)
            f_minus = self.f(q_iminus)

            self.J[:, i] = (f_plus - f_minus) / self.delta_q[i]

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

    uz_0 = np.array([[0, 0, 0]]).transpose()
    q = np.array([0, 0, 0, 0, np.pi, 0])  # inputs q
    delta_q = np.ones(6) * 1e-3
    x = np.zeros(3)

    r_jac = Jacobian(delta_q, x, q, uz_0, model)
    # r_jac.jac_approx()
    J = r_jac.J
    J_inv = r_jac.p_inv()

    print('J:\n', J)
    print('\nJ_inv:\n', J_inv)
    print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(J_inv, J))))
    print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv, np.dot(J_inv, np.dot(J, J_inv))))

    # q = J_inv * 