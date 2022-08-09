import matplotlib.pyplot as plt
import numpy as np
import utils
import sys
import os

from pathlib import Path


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_consensus(res_dir: Path):
    vnames = {'momentum': r'$\sum_{i=1}^{N_e}\frac{1}{N_e}\|\bar{p}-p_0^i\|_{L^2}$',
              'theta': r'$\sum_{i=1}^{N_e}\frac{1}{N_e}\|\bar{\theta}-\theta^i\|_{L^2}$'}

    for var in ['momentum', 'theta']:
        if os.path.isfile(res_dir / 'consensuses_{}'.format(var)):
            consensuses = utils.pload(res_dir / 'consensuses_{}'.format(var))

            # plot pickles
            print("Plotting momentum consensuses...")
            ax = plt.subplot(111)
            ax.set_xlim(left=1, right=len(consensuses))
            plt.grid()
            plt.xlabel('Iteration $n$')
            plt.ylabel(vnames[var])
            plt.plot(range(1, len(consensuses) + 1), consensuses)
            plt.savefig(res_dir / 'consensuses_{}.pdf'.format(var))
            plt.clf()


def plot_errors(res_dir: Path):
    eta = -1
    params_path = res_dir / 'inverse_problem_parameters.log'
    if os.path.isfile(params_path):
        params = open(params_path, "r")
        for line in params:
            if line.startswith("eta:"):
                eta = float(line.split()[1])
                break

    errors_path = res_dir / 'errors'
    if os.path.isfile(errors_path):
        print(r'Plotting errors...')
        errors = utils.pload(errors_path)
        plt.grid()
        plt.xlabel('Iteration $n$')
        plt.ylabel(r'Error')
        plt.plot(range(1, len(errors) + 1), errors, label=r'Error norm')
        plt.axhline(y=eta, linestyle=':', color='red', label=r'Noise level')
        plt.legend(loc='best')
        plt.savefig(res_dir / 'errors.pdf')
        plt.clf()


def plot_shape_means(res_dir: Path):
    # count pickles
    num_q_means = 0
    prefix = "q_mean_iter="
    for fn in os.listdir(res_dir):
        if fn.startswith(prefix):
            num_q_means += 1
    print("Number of q_mean files:", num_q_means)

    # load pickles
    q_means = []
    for i in range(num_q_means):
        q_mean = utils.pload(res_dir / (prefix + str(i)))
        q_means.append(q_mean)

    # plot pickles
    print("Plotting means...")
    plt.grid()
    plt.xlabel('$x$-coordinate')
    plt.ylabel('$y$-coordinate')
    j = 0
    while j < num_q_means:
        shape = np.append(q_means[j], [q_means[j][0, :]], axis=0)
        plt.plot(shape[:, 0], shape[:, 1])
        j += print_freq

    target = utils.pload(res_dir / "target")
    target = np.append(target, [target[0, :]], axis=0)
    plt.plot(target[:, 0], target[:, 1], 'd:', label='Target')
    plt.legend(loc='best')
    plt.savefig(res_dir / 'q_means.pdf')
    plt.clf()


def plot_t_means(res_dir: Path):

    # count pickles
    num_t_means = 0
    prefix = "t_mean_iter="
    for fn in os.listdir(res_dir):
        if fn.startswith(prefix):
            num_t_means += 1
    print("Number of t_mean files:", num_t_means)

    # load pickles
    t_means = []
    for i in range(num_t_means):
        t_mean = utils.pload(res_dir / (prefix + str(i)))
        t_means.append(t_mean)

    # plot pickles
    print("Plotting theta means...")
    plt.grid()
    plt.xlabel('$x$-coordinate')
    plt.ylabel('$y$-coordinate')
    circ = lambda t: np.array([np.cos(t), np.sin(t)])
    j = 0
    while j < num_t_means:
        t_means_shape = np.array(list(map(circ, t_means[j])))
        shape = np.append(t_means_shape, [t_means_shape[0, :]], axis=0)
        plt.plot(shape[:, 0], shape[:, 1])
        j += print_freq

    truth_path = res_dir / "t_truth"

    if truth_path.exists():
        t_truth = utils.pload(truth_path)
        t_truth = np.array(list(map(circ, t_truth)))
        t_truth = np.append(t_truth, [t_truth[0, :]], axis=0)
        plt.plot(t_truth[:, 0], t_truth[:, 1], 'd:')

    plt.savefig(res_dir / 't_means.pdf')
    plt.clf()


def plot_alphas(res_dir: Path):
    for suffix, tex in [('momentum', '\\mathbf{P}'), ('thetas', '\\mathbf{\\Theta}')]:
        if os.path.isfile(res_dir / 'alphas'):
            print(r'Plotting $\alphas_{}$...'.format(suffix))
            alphas = utils.pload(res_dir / 'alphas')
            plt.grid()
            plt.xlabel('Iteration $n$')
            plt.ylabel(r'$\ln(\alpha_n^{})$'.format(tex))
            plt.semilogy(range(1, len(alphas) + 1), alphas)
            plt.savefig(res_dir / 'alphas_{}.pdf'.format(suffix))
            plt.clf()


if __name__ == "__main__":
    res_dir = Path(sys.argv[1])
    num_files = len(next(os.walk(res_dir))[2])
    if len(sys.argv) >= 3:
        print_freq = int(sys.argv[2])
    else:
        print_freq = 1

    plot_consensus(res_dir)
    plot_errors(res_dir)
    plot_shape_means(res_dir)
    plot_t_means(res_dir)
