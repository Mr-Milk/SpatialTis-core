import somoclu
from scipy.stats import chi2

from .util import *


def Sparun(X, exp_tab, kernel_space=None):
    # Perform SpatialDE test
    if kernel_space == None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            'const': 0
        }

    results = dyn_de(X, exp_tab, kernel_space)
    mll_results = get_mll_results(results)

    # Perform significance test
    mll_results['pval'] = 1 - chi2.cdf(mll_results['LLR'], df=1)
    mll_results['qval'] = qvalue(mll_results['pval'])

    return mll_results


class SomNode:
    def __init__(self, X, k, homogeneous_codebook=True):
        self.X = X
        self.somn = int(np.sqrt(X.shape[0] // k))
        self.ndf = None
        self.ninfo = None
        self.nres = None
        if homogeneous_codebook:
            xmin, ymin = X.min(0)
            xmax, ymax = X.max(0)
            cobx, coby = np.meshgrid(np.linspace(xmin, xmax, self.somn), np.linspace(ymin, ymax, self.somn))
            self.inicodebook = np.transpose(np.array([cobx.ravel(), coby.ravel()], np.float32), (1, 0))
            self.som = somoclu.Somoclu(self.somn, self.somn, initialcodebook=self.inicodebook.copy())
        else:
            self.som = somoclu.Somoclu(self.somn, self.somn)
        self.som.train(X, epochs=10)

    def reTrain(self, ep):
        self.som.train(self.X, epochs=ep)

    def mtx(self, df, alpha=0.5):
        bsmc = self.som.bmus
        soml = []
        for i in np.arange(bsmc.shape[0]):
            u, v = bsmc[i]
            soml.append(v * self.somn + u)
        ndf_value = np.zeros([df.shape[0], len(np.unique(np.array(soml)))])
        ninfo = pd.DataFrame(columns=['x', 'y', 'total_count'])
        tmp = 0
        for i in np.unique(np.array(soml)):
            select_df = df.loc[:, np.array(soml) == i]
            ndf_value[:, tmp] = alpha * select_df.max(1) + (1 - alpha) * select_df.mean(1)
            coor = self.som.codebook[i // self.somn, i % self.somn]
            ninfo.loc[tmp, 'x'] = round(coor[0], 4)
            ninfo.loc[tmp, 'y'] = round(coor[1], 4)
            tmp += 1
        ndf = pd.DataFrame(ndf_value, index=list(df.T))
        ninfo.total_count = ndf.sum(0)
        self.ndf = ndf
        self.ninfo = ninfo
        return ndf, ninfo,

    def norm(self):
        if self.ninfo is None:
            raise ValueError('please generate mtx first')
        dfm = stabilize(self.ndf)
        self.nres = regress_out(self.ninfo, dfm, 'np.log(total_count)').T
        return self.nres

    def run(self):
        if self.nres is None:
            self.norm()
        X = self.ninfo[['x', 'y']].values.astype(float)
        result = Sparun(X, self.nres)
        result.sort_values('LLR', inplace=True, ascending=False)
        number_q = result[result.qval < 0.05].shape[0]
        return result, number_q
