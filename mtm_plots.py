# type: ignore
import math
import os
import torch
import torch.nn.functional as F
import math
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from enum import StrEnum
from resources.tropnis import SampleBatch, TriangleIntegrand, TropicalFlow, Cache
from triangle import ltd_triangle, prop_factor


# Integrand functions
def const_f(m_psi: float, k: list[float], q: list[float], p: list[float], weight: float) -> float:
    return prop_factor(m_psi, k, q, p, weight)


def triangle_f(m_psi: float, k: list[float], q: list[float], p: list[float], weight: float) -> float:
    return ltd_triangle(m_psi, k, q, p, weight) / 8.


# Logging related setup
class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


logging.basicConfig(
    format=f'{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{
        Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
logger = logging.getLogger('mtm_plots')


class PlottingBatch:
    def __init__(self, grid_slice: np.ndarray, sampleBatch: SampleBatch):
        self.grid_slice = grid_slice
        self.func_slice = []
        self.prob_slice = []
        self.prob_channel = []
        self.y_channel = []
        self.loss = []
        self.samples = sampleBatch


class TropicalIntegrator:
    n_slice = 1000
    plt_samples = 10000

    def __init__(self, integrand, lr=3e-4, batch_size=1024, continuous_kwargs={}, discrete_kwargs={}):
        self.integrand = integrand
        self.flow = TropicalFlow(
            continuous_dim=integrand.continuous_dim,
            discrete_dims=integrand.discrete_dims,
            conditional_dim=0,
            continuous_kwargs=continuous_kwargs,
            discrete_kwargs=discrete_kwargs,
        )
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr)
        self.batch_size = batch_size
        self.channels = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
        self.plottables = PlottingBatch(
            self.gen_1dSlice(), self.sample(self.plt_samples))
        self.n_train = 0
        self.loss = 0.
        self.loss_plt = 0.
        self.update_plottables_channel()
        self.update_plottables_slice()
        # Horrible hack job, this is so we can access all the individual sample data as an ordered tensor
        self.plottables.samples = SampleBatch(
            self.plottables.samples.x.unsqueeze(0),
            self.plottables.samples.indices.unsqueeze(0),
            self.plottables.samples.prob.unsqueeze(0),
            self.plottables.samples.discrete_probs.unsqueeze(0),
            self.plottables.samples.func_val.unsqueeze(0)
        )

    def sample(self, n: int) -> SampleBatch:
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                indices[:, i], cache = self.flow.sample_discrete(
                    i, pred_probs, cache)

            x, log_prob = self.flow.sample_continuous(cache)
            func_val = self.integrand(indices, x)
            return SampleBatch(
                x, indices, log_prob.exp(), torch.cat(discrete_probs, dim=1), func_val
            )

    def optimization_step(self, samples: SampleBatch) -> float:
        self.optimizer.zero_grad()
        log_prob = self.flow.log_prob(
            samples.indices, samples.x, samples.discrete_probs)
        loss = -torch.mean(samples.func_val.abs() / samples.prob * log_prob)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, iterations: int, n_log=50, n_plot=10) -> None:
        for it in range(self.n_train, self.n_train + iterations):
            samples = self.sample(self.batch_size)
            it_loss = self.optimization_step(samples)
            self.loss += it_loss
            self.loss_plt += it_loss
            if (it + 1) % n_plot == 0:
                self.update_plottables_channel()
                self.plottables.loss.append(self.loss_plt/n_plot)
                self.loss_plt = 0.
            if (it + 1) % n_log == 0:
                logger.info(
                    f"Training iteration {it+1}: loss={self.loss / n_log:.6f}")
                self.loss = 0.
        self.update_plottables_slice()
        self.update_plottables_samples()
        self.n_train += iterations

    def update_plottables_channel(self) -> None:
        prob, y_exp = self.peek_discrete_channels()
        self.plottables.prob_channel.append(prob.numpy())
        self.plottables.y_channel.append(y_exp.numpy())

    def update_plottables_slice(self) -> None:
        func_slice = []
        prob_slice = []

        for ch in self.channels:
            slice_samples = self.sample_1dslice(ch, self.plottables.grid_slice)
            func_slice.append(slice_samples.func_val)
            prob_slice.append(slice_samples.prob)

        self.plottables.func_slice.append(func_slice)
        self.plottables.prob_slice.append(prob_slice)

    # Horrible hack job, this is so we can access all the individual sample data as an ordered tensor
    def update_plottables_samples(self) -> None:
        samples = self.sample(self.plt_samples)
        x = torch.cat([self.plottables.samples.x, samples.x.unsqueeze(0)])
        indices = torch.cat([self.plottables.samples.indices,
                            samples.indices.unsqueeze(0)])
        prob = torch.cat([self.plottables.samples.prob,
                         samples.prob.unsqueeze(0)])
        discrete_probs = torch.cat(
            [self.plottables.samples.discrete_probs, samples.discrete_probs.unsqueeze(0)])
        func_val = torch.cat([self.plottables.samples.func_val,
                             samples.func_val.unsqueeze(0)])

        self.plottables.samples = SampleBatch(
            x, indices, prob, discrete_probs, func_val)

    def integrate(self, n: int) -> tuple[float, float]:
        samples = self.sample(n)
        weights = samples.func_val / samples.prob
        integral = weights.mean().item()
        error = weights.std().item() / math.sqrt(n)
        return integral, error

    def peek_discrete_channels(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            n = sum(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for dim in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    dim, indices[:, :dim])
                discrete_probs.append(pred_probs)
                x, prob, net_cache = cache
                y, net_cache = self.flow.masked_net.forward_cached(
                    x, dim, net_cache)
                unnorm_probs = y.exp() * pred_probs
                cdf = unnorm_probs.cumsum(dim=1)
                norm = cdf[:, -1]
                cdf = cdf / norm[:, None]
                if dim == 0:
                    samples = torch.tensor([0, 0, 1, 1, 2, 2])
                elif dim == 1:
                    samples = torch.tensor([1, 2, 0, 2, 0, 1])
                else:
                    logger.critical(
                        "The discrete channel peeker entered an unintended state, dim>2.")

                y_exp = torch.gather(unnorm_probs, 1, samples[:, None])[
                    :, 0] / norm
                prob = prob * y_exp
                x_one_hot = F.one_hot(
                    samples, self.integrand.discrete_dims[dim]).to(y.dtype)
                indices[:, dim], cache = samples, (x_one_hot, prob, net_cache)

            return prob, y_exp*2.

    def sample_1dslice(self, channel, force_z) -> SampleBatch:
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros(
                (self.n_slice, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(self.n_slice)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                indices[:, i], cache = self.sample_discrete_1dslice(
                    i, pred_probs, cache, channel)

            x, log_prob = self.sample_continuous_1dslice(cache, force_z)
            func_val = self.integrand(indices, x)
            return SampleBatch(
                x, indices, log_prob.exp(), torch.cat(discrete_probs, dim=1), func_val
            )

    def sample_discrete_1dslice(
        self, dim: int, pred_probs: torch.Tensor, cache: Cache, channel: list[int]
    ) -> Cache:
        x, prob, net_cache = cache
        y, net_cache = self.flow.masked_net.forward_cached(x, dim, net_cache)
        unnorm_probs = y.exp() * pred_probs
        cdf = unnorm_probs.cumsum(dim=1)
        norm = cdf[:, -1]
        cdf = cdf / norm[:, None]
        samples = torch.tensor([channel[dim]]).repeat(y.shape[0])
        prob = torch.gather(unnorm_probs, 1, samples[:, None])[
            :, 0] / norm * prob
        x_one_hot = F.one_hot(
            samples, self.integrand.discrete_dims[dim]).to(y.dtype)
        return samples, (x_one_hot, prob, net_cache)

    def sample_continuous_1dslice(self, cache: Cache, force_z) -> tuple[torch.Tensor, torch.Tensor]:
        x, prob, net_cache = cache
        condition = torch.cat((net_cache[0], x), dim=1)
        flow_samples, jac = self.flow.flow.transform(
            x=force_z, inverse=True, c=condition)
        log_prob_latent = self.flow.flow._latent_log_prob(force_z)
        flow_log_prob = log_prob_latent - jac
        return flow_samples, prob.log() + flow_log_prob

    def gen_1dSlice(self) -> torch.Tensor:
        n_dim = self.integrand.continuous_dim
        # generate slice from random point p_0 and random direction p_dir in continuous x_space
        p_0 = F.normalize(torch.rand(n_dim), dim=0)
        p_dir = F.normalize(torch.rand(n_dim), dim=0)
        # for what values of t does x(t) = p_0 + t*p_dir intersect the [0,1] hypercube boundary?
        min_t = (p_0/p_dir).min().item()
        max_t = ((1-p_0)/p_dir).min().item()
        delta_t = (max_t - min_t)/(self.n_slice - 1)
        # interpolate between the intersection points
        grid_1dslice = torch.empty((self.n_slice, n_dim))
        for i in range(self.n_slice):
            grid_1dslice[i] = p_0 + (min_t + i*delta_t)*p_dir
        return grid_1dslice


def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='mtm_plots')

    parser.add_argument('-p', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, 0.005],
                        help='Four-momentum of the first photon. Default = %(default)s GeV')
    parser.add_argument('-q', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, -0.005],
                        help='Four-momentum of the second photon. Default = %(default)s GeV')
    parser.add_argument('--m_psi', type=float,
                        default=0.02,
                        help='Mass of the spinors. Default = %(default)s GeV')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Plot all figures')
    parser.add_argument('--slice', action='store_true', default=False,
                        help='Plot the 1D-slice figures')
    parser.add_argument('--samples_prog', action='store_true', default=False,
                        help='Plot the distribution of sampled weights')
    parser.add_argument('--channel_prog', action='store_true', default=False,
                        help='Plot the discrete propability progression')
    parser.add_argument('--n_samples', '-ns', type=int, default=10000,
                        help='Set number of samples taken from trained model. Default = %(default)s')
    parser.add_argument('--n_iterations', '-ni', nargs='+', type=int,
                        help='Set number of training iterations as a list, after each entry an 1dslice sample will be taken')
    parser.add_argument('--batch_size', '-bs', type=int, default=1024,
                        help='Set batch size per training iteration')
    parser.add_argument('--log_interval', '-li', type=int, default=20,
                        help='Set number of batches between logging')
    parser.add_argument('--plotting_interval', '-pi', type=int, default=2,
                        help='Set the number of batches between plotting the data. Default = %(default)s')
    parser.add_argument('--verbosity', '-v', type=str, choices=[
                        'debug', 'info', 'critical'], default='info', help='Set verbosity level of the logger.')
    parser.add_argument('--nosave_fig', action='store_true', default=False,
                        help='Enable to only show plotted figures.')
    parser.add_argument('--file_path', '-fp', type=str, default='',
                        help='Specify a relative path to a folder to save the plot at.')

    args = parser.parse_args()

    plot_1dslice = args.all or args.slice
    plot_samples = args.all or args.samples_prog
    plot_channel_prog = args.all or args.channel_prog

    match args.verbosity:
        case 'debug': logger.setLevel(logging.DEBUG)
        case 'info': logger.setLevel(logging.INFO)
        case 'critical': logger.setLevel(logging.CRITICAL)

    subfolder_path = os.path.join(os.getcwd(), args.file_path)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        logger.info(
            f"Created target folder {subfolder_path} for plotting results, please make sure this is the intended location.")

    torch.set_default_dtype(torch.float64)
    integrand = TriangleIntegrand(triangle_f, args.m_psi, args.p, args.q)
    integrator = TropicalIntegrator(integrand, batch_size=args.batch_size)

    for n_iteration in args.n_iterations:
        logger.info(f"Running training batch: {n_iteration} iterations")
        integrator.train(n_iteration, args.log_interval,
                         args.plotting_interval)
    logger.info("Training done \n")

    int_flow, err_flow = integrator.integrate(args.n_samples)
    rsd_flow = err_flow / int_flow * math.sqrt(args.n_samples)
    logger.info(f"Trained flow:     {
        int_flow:.8f} +- {err_flow:.8f}, RSD = {rsd_flow:.2f}")

    n_train_its = [0]
    for n_iteration in args.n_iterations:
        n_train_its.append(n_train_its[-1] + n_iteration)
    n_train_batches = len(n_train_its)
    train_its_lbls = [f'it: {it}' for it in n_train_its]

    channel_sigs = [
        f'{ch[0]+1}{ch[1]+1}' for ch in integrator.channels]
    n_channels = len(integrator.channels)

    plt.style.use('ggplot')

    if plot_samples:
        # Histogram of the sample weight distribution for each channel
        n_bins = 30
        unique_ind = torch.tensor(integrator.channels)
        samples = integrator.plottables.samples
        # This mask is used to divide the samples into the sectors "channels" they fall into
        # It will have shape (#channels, #training_batches, #samples)
        ch_mask = [torch.all(torch.eq(
            samples.indices, ind), dim=2) for ind in unique_ind]
        ch_mask = torch.stack(ch_mask, dim=0)

        all_weights = samples.func_val / samples.prob
        min_weight = all_weights.min().item()
        max_weight = all_weights.max().item()
        logbins = np.geomspace(min_weight, max_weight, n_bins+1)
        fig, axs = plt.subplots(
            2, 3, sharex=True, sharey=True, layout='constrained')
        for ch in range(n_channels):
            ax = axs[ch % 2, ch//2]
            mask = ch_mask[ch]
            weights = [all_weights[it, mask[it, :]].numpy()
                       for it in range(len(n_train_its))]
            ax.hist(weights, bins=logbins, histtype='step', alpha=0.7,
                    linewidth=1, label=train_its_lbls)
            ax.set_xscale('log')
            ax.set_yscale('log')
            if ch % 2 == 1:
                ax.set_xlabel('weight')
            if ch // 2 == 0:
                ax.set_ylabel('# samples')
            ax.set_title(f'ch{channel_sigs[ch]}',
                         va='top', ha='left', x=0.1, y=0.9)

        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc='outside upper center',
                   ncol=n_train_batches)
        if not args.nosave_fig:
            filename = 'sample_weight_evolution_' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), args.file_path,
                        filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()

        # Box plot of the sample weights for each channel, ordered by std after training
        after_weights = [all_weights[-1, ch_mask[ch, -1, :]]
                         for ch in range(n_channels)]
        after_rsd = [w.std().item() / w.mean().item() for w in after_weights]

        sorted_by_rsd = sorted(enumerate(after_rsd), key=lambda x: x[-1])
        sorted_ind = [s[0] for s in sorted_by_rsd]

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        for it in range(n_train_batches):
            weights = [all_weights[it, ch_mask[ch, it, :]].numpy()
                       for ch in sorted_ind]
            means = [w.mean() for w in weights]
            qtls = [np.quantile(w, [0.1, 0.9]) for w in weights]
            box_positions = np.arange(
                it+1, (n_train_batches+1)*n_channels+1, n_train_batches+1)

            ax.boxplot(weights, conf_intervals=qtls, positions=box_positions, usermedians=means,
                       orientation='horizontal', showfliers=False, whis=0)
        ax.set_xscale('log')
        tick_positions = np.arange(
            (n_train_batches+1)/2, (n_train_batches+1)*n_channels+1, n_train_batches+1)
        tick_labels = [f'ch{channel_sigs[ind]}' for ind in sorted_ind]
        ax.set_yticks(tick_positions, tick_labels)
        ax.grid(False)
        ax.invert_yaxis()
        if not args.nosave_fig:
            filename = 'ch_summ' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), args.file_path,
                        filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()

    if plot_channel_prog:
        # Plot the channel probability progression
        legend_loc = 'upper right'
        probs = np.array(integrator.plottables.prob_channel)
        losses = np.array(integrator.plottables.loss)
        loss_scaling = np.max(probs)/np.max(losses)
        losses *= loss_scaling
        log_iters = np.arange(start=0,
                              stop=integrator.n_train+1, step=args.plotting_interval)

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle(
            f'Channel Probability Evolution using {args.batch_size} samples')

        for ch in range(3):
            channel = probs[:, 2*ch] + probs[:, 2*ch+1]
            lbl = f'Ch{ch+1}'
            ax = axs[ch % 2, ch//2]
            ax.plot(log_iters, channel, label=lbl+'x')
            ax.plot(log_iters, probs[:, 2*ch],
                    label=f'Ch{channel_sigs[2*ch]}')
            ax.plot(log_iters, probs[:, 2*ch+1],
                    label=f'Ch{channel_sigs[2*ch+1]}')
            ax.set_title(lbl)
            ax.legend(loc=legend_loc)
            axs[1, 1].plot(log_iters, channel, label=lbl+'x')

        axs[1, 1].plot(log_iters[1:], losses,
                       color='black', label=f'{loss_scaling:.1f}*L')
        axs[1, 1].legend(loc=legend_loc)
        axs[1, 1].set_title('Summary')

        plt.ylim([0, np.ceil(np.max(probs)*20)/10])
        if not args.nosave_fig:
            filename = 'channel_probs_evolution_' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), args.file_path,
                        filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()

        # Plot the progression of the derivative of the
        log_iters = np.arange(start=0,
                              stop=integrator.n_train, step=args.plotting_interval)
        fig, ax = plt.subplots()
        fig.suptitle(
            f'Channel probdiff Evolution using {args.batch_size} samples')
        cols = ['red', 'blue', 'green']
        for ch in range(3):
            prob_diffs = (probs[1:, 2*ch] / probs[:-1, 2*ch]
                          )**(1/args.plotting_interval)
            ax.plot(log_iters, prob_diffs,
                    label=f'Ch{channel_sigs[2*ch]}', ls='-', color=cols[ch])
            prob_diffs = (probs[1:, 2*ch+1] / probs[:-1, 2*ch+1]
                          )**(1/args.plotting_interval)
            ax.plot(log_iters, prob_diffs,
                    label=f'Ch{channel_sigs[2*ch+1]}', ls='--', color=cols[ch])
        ax.legend(loc=legend_loc)

        if not args.nosave_fig:
            filename = 'channel_dprobs_evolution_' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), args.file_path,
                        filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()

    if plot_1dslice:
        # Weight sampled across a 1D slice in continuum space, for each channel
        t_grid = np.linspace(0, 1, integrator.n_slice)
        func_vals = integrator.plottables.func_slice
        probs = integrator.plottables.prob_slice

        fig, axs = plt.subplots(
            2, 3, sharex=True, sharey=True, layout='constrained')
        for ch in range(n_channels):
            weights = [(f[ch] / p[ch]).numpy()
                       for f, p in zip(func_vals, probs)]
            ax = axs[ch % 2, ch//2]
            for weight, lbl in zip(weights, train_its_lbls):
                ax.plot(t_grid, weight, linewidth=1, label=lbl)
            if ch % 2 == 1:
                ax.set_xlabel('t')
            if ch // 2 == 0:
                ax.set_ylabel('weight')
            ax.set_title(f'ch{channel_sigs[ch]}',
                         va='top', ha='left', x=0.1, y=0.9)
            ax.set_yscale('log')

        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc='outside upper center',
                   ncol=n_train_batches)
        if not args.nosave_fig:
            filename = 'slice_weight_evolution' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), args.file_path,
                        filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()

    if args.all:
        # Plot the 1dslice in continuous parameter space, func_val and weight too (OLD)
        cols = iter(["RED", "BLUE", "GREEN", "BLACK"])
        t_grid = np.linspace(0, 1, integrator.n_slice)

        for func, prob, train_it in zip(integrator.plottables.func_slice, integrator.plottables.prob_slice, n_train_its):
            weight = func[0]/prob[0]
            fig, ax = plt.subplots()
            fig.suptitle(f"1dSlice after {train_it} training steps")
            ax.plot(t_grid, func[0].numpy(), label='f')
            ax.plot(t_grid, prob[0].numpy(), label='p')
            ax.plot(t_grid, weight.numpy(), label='w')
            ax.hlines(weight.mean().item(), t_grid.min(), t_grid.max(),
                      colors="black", linestyles='dashed')
            ax.set_xlabel("t")
            ax.set_ylabel("value")
            ax.set_yscale('log')
            ax.legend()
            if not args.nosave_fig:
                filename = 'slice_old' + \
                    datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
                plt.savefig(os.path.join(os.getcwd(), args.file_path,
                                         filename), dpi=300, bbox_inches='tight')
            else:
                plt.show()


if __name__ == "__main__":
    main()
