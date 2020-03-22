import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from config import *
import cvxpy as cp


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def get_hs_bound(self, norm_bound:float, beta:float):
        """
            --------------------------------------------------------------
            Compute Hockey stick divergence upper bound, given norm bound, 
            under gaussian noise smoothing.
            --------------------------------------------------------------
            norm_bound : two-norm bound for |x-x'|
            sigma : standard deviation for Gaussian noise
            beta : parameter for HS divergence
            --------------------------------------------------------------
            HS_bound formulation (Could be found in Appendix 7.2, Lemma 6) : 

                max_{x': |x-x'|_2 <= epsilon} D_{HS,beta}[u(x')||u(x)]
                = Phi( epsilon/(2*sigma)-log(beta)*sigma/(2*epsilon) )
                - beta*Phi( -epsilon/(2*sigma) - log(beta)*sigma/(2*epsilon) )
                - max(1-beta,0)

                Note that, Phi is the CDF of a standard Gaussain variable.
            --------------------------------------------------------------
        """
        sigma = self.sigma
        a = norm_bound/(2*sigma) - np.log(beta)*sigma/(2*norm_bound)
        b = -norm_bound/(2*sigma) - np.log(beta)*sigma/(2*norm_bound)
        HS_bound = norm.cdf(a)-beta*norm.cdf(b)-max(1-beta,0)
        return max(0,HS_bound)

    def optimizer(self, epsilon_0: float, epsilon_1: float, beta_0 : float, beta_1 : float,\
        pa: float, pb: float, pc: float) -> float :
        """
            Lagrange Dual : 

                lambda_0*[pa*f_0(ra)+pb*f_0(rb)+pc*f_0(rc)-epsilon_0] + 
                lambda_1*[pa*f_1(ra)+pb*f_1(rb)+pc*f_1(rc)-epsilon_1] +
                k*(1-pa*ra-pb*rb-pc*rc) + pa*ra - pb*rb 

        """
        print("p : ",pa,pb,pc)
        ##### ra, rb, rc #########
        ra = cp.Variable()
        rb = cp.Variable()
        rc = cp.Variable()
        ##########################

        ####### f-function ######################################
        f_0_ra = cp.maximum(ra-beta_0,0) - cp.maximum(1-beta_0,0)
        f_0_rb = cp.maximum(rb-beta_0,0) - cp.maximum(1-beta_0,0)
        f_0_rc = cp.maximum(rc-beta_0,0) - cp.maximum(1-beta_0,0)
        
        f_1_ra = cp.maximum(ra-beta_1,0) - cp.maximum(1-beta_1,0)
        f_1_rb = cp.maximum(rb-beta_1,0) - cp.maximum(1-beta_1,0)
        f_1_rc = cp.maximum(rc-beta_1,0) - cp.maximum(1-beta_1,0)
        #########################################################

        ####### lambda_0, lambda_1, k ###############
        lambda_0 = 0.0
        lambda_1 = 0.0
        k = 0.0
        #############################################

    

        
        n_iter = 50
        step_size = 0.1
        end_point = 0.001

        for i in range(n_iter):

            Lagrange_aug = lambda_0*(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0) + \
                lambda_1*(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1) + \
                k*(1-pa*ra-pb*rb-pc*rc) + pa*ra - pb*rb + cp.square(1-pa*ra-pb*rb-pc*rc) + \
                cp.square(cp.maximum(0,pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1)) + \
                cp.square(cp.maximum(0,pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0))
            
            cp.Problem(cp.Minimize(Lagrange_aug)).solve(solver = cp.ECOS)
            
            o_lambda_0 = lambda_0
            lambda_0 += step_size*(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0).value
            lambda_0 = max(lambda_0,0)

            o_lambda_1 = lambda_1
            lambda_1 += step_size*(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1).value
            lambda_1 = max(lambda_1,0)

            o_k = k
            k += step_size*(1-pa*ra-pb*rb-pc*rc).value

            if abs(o_lambda_0-lambda_0)<end_point and abs(o_lambda_1-lambda_1)<end_point \
            and abs(o_k-k)<end_point :
                break
        
        """ debug
        print("lambda_0 = ",lambda_0)
        print("pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0 = ",(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0).value)
        print("lambda_1 = ",lambda_1)
        print("pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1 = ",(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1).value)
        print("k = ",k)
        print("1-pa*ra-pb*rb-pc*rc",(1-pa*ra-pb*rb-pc*rc).value)
        """
        return (pa*ra - pb*rb).value





    def certify(self, x: torch.tensor, true_label, n0: int, n: int, alpha: float,\
         batch_size: int, radius: float = 0.5, full_information: bool = False) -> (bool):
        """
            classifier : h ; true_label : c
            # note that radius is under the 2-norm measure

            ------------- phi function -----------------------
            pair wise phi function : phi_{c,c'} 
            phi_{c,c'}(x') only has three possible outputs :  {-1,0,1}
            phi = 1 , when h(x')=c
            phi = -1 , when h(x')=c'
            phi = 0 , otherwise

            full information robust <=> E(phi)>=0 within the pertubation radius for each pair of (c,c')
            limited information robust :
                phi = 1 , when h(x')=c
                phi = -1 , otherwise
            --------------------------------------------------

            ------------ idea of f-divergence certification ----------
            norm-bound ---> relaxation ---> divergence bound
            ----------------------------------------------------------

            ------------- dual target --------------------
            Let :
                P(phi=1) = pa
                P(phi=-1) = pb
                P(phi=0) = pc
            
            r(x) = v(x) / rho(x)   ; v is the distrubed distribution , rho is the original distribution
            
            Let :
                ra for phi=1
                rb for phi=-1
                rc for phi=0
            
            epsilon : divergence bound

            f : f-function for divergence metric

            max_{lambda,k} 
            < min_{r}   k - lambda*epsilon - pa*[ra*(k-1)-lambda*f(ra)] 
                            - pb*[rb*(k+1)-lambda*f(rb)] - pc*[rc*k-lambda*f(rc)] >
                <=>
            max_{lambda,k} 
            < min_{r}   lambda*[pa*f(ra)+pb*f(rb)+pc*f(rc)-epsilon] 
                            + k*(1-pa*ra-pb*rb-pc*rc) + pa*ra - pb*rb >
            --------------------------------------------------

            -------- algorithm -------
                n0 sampling for estimating pa,pb,pc => computing approximate lambda and k
                n1 sampling  to estimate the lowerbound of the target funtion (using approximate lambda and k as setting)
            --------------------------
        
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x,n0,batch_size)
        top = np.argmax(counts)
        
        if top!=true_label:
            return False

        # By defualt, we use two Hockey-stick divergence for certificaiton.
        # It's recommended to use two HS divergences. 
        # As the original work doesn't give out the method to finetune beta parameters for HS divergence,
        # emperically, we use beta = 0.7 and beta = 2.5 for the two HS divergences respectively.  

        # bound for beta=2
        beta_0 = 2.5
        epsilon_0 = self.get_hs_bound(norm_bound=radius,beta=beta_0)
        # bound for beta=8
        beta_1 = 0.7
        epsilon_1 = self.get_hs_bound(norm_bound=radius,beta=beta_1)

        #   def optimizer(self, epsilon_0: float, epsilon_1: float, beta_0 : float, beta_1 : float,\
        #                       pa: float, pb: float, pc: float) -> float :


        ############# note that #############
        """
            In standard algorithm, n0 for estimating lambda and k
            n1 for estimating the value of Lagrange , and with a confidence alpha...
            Here, for efficiency of our rough experiment, we only perform n0 sampling...
            This will lead to a slight reduciton of the reliability of the result...
            But empeirically, the difference won't be larage...
        """
        #####################################
        if full_information :
            for i in range(n_class):
                if i==top : continue
                pa = counts[top]/n0
                pb = counts[i]/n0
                pc = max(0,1-pa-pb) # avoiding small negative due to numeric error 
                result = self.optimizer(epsilon_0,epsilon_1,beta_0,beta_1,pa,pb,pc)
                if result<0:
                    return False
            return True
        else:
            pa = counts[top]/n0
            if pa<0.5:
                return False
            pb = 1-pa
            pc = 0
            result = self.optimizer(epsilon_0,epsilon_1,beta_0,beta_1,pa,pb,pc)
            if result<0:
                return False
            else:
                return True

    def predict(self, x: torch.tensor, n: int, batch_size: int,\
        enable_abstain: bool = False,alpha: float = None) -> int:
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)

        if enable_abstain and alpha is not None :
            top2 = counts.argsort()[::-1][:2]
            count1 = counts[top2[0]]
            count2 = counts[top2[1]]
            if binom_test(count1, count1 + count2, p=0.5) > alpha:
                return Smooth.ABSTAIN
            else:
                return top2[0]
        else:
            top = np.argmax(counts)
            return top

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch) * self.sigma
                samples = (batch+noise).to(device)

                predictions = self.base_classifier(samples).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts


