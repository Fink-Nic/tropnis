# type: ignore
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import momtrop
import pydot
import json
import toml
import os
import numpy as np
from gammaloop import GammaLoopAPI
from resources.helpers import PATHS, deep_update


@dataclass
class GraphProperties:
    edge_src_dst_vertices: list[(int, int)]
    edge_masses: list[int]
    edge_momentum_shifts: list[list[float]]
    graph_external_vertices: list[int]
    graph_signature: list[list[int]]
    momtrop_edge_weights: list[int] = []

    def __post_init__(self):
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]


class ModelParser:
    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(self.model_path, 'r') as f:
            self.model = json.load(f)

    def get_particle_from_identifier(self, identifier_name: str, value) -> Dict:
        particle_match = None
        for particle in self.model['particles']:
            try:
                if particle[identifier_name] == value:
                    particle_match = particle
            except:
                pass

        if particle_match is None:
            raise KeyError(
                f"Particle with {identifier_name}='{value}' does not exist in model '{self.model_path}'.")

        return particle_match

    def get_particle_parameter_from_identifier(self,
                                               identifier_name: str,
                                               identifier_value,
                                               parameter_name: str):
        particle_match = self.get_particle_from_identifier(
            identifier_name, identifier_value)
        try:
            model_parameter_name = particle_match[parameter_name]
        except KeyError:
            raise KeyError(
                f"Particle with '{identifier_name}' = '{identifier_value}' "
                + f"does not have parameter '{parameter_name}'.")

        if model_parameter_name == 'ZERO':
            return [0., 0.]

        parameter_match = None
        for parameter in self.model['parameters']:
            try:
                if parameter['name'] == model_parameter_name:
                    parameter_match = parameter['value']
            except:
                pass

        if parameter_match is None:
            raise KeyError(f"The model '{self.model_path}' does not specify a value for "
                           + f"the parameter '{model_parameter_name}'.")

        return parameter_match

    def get_particle_parameter_from_name(self, particle_name: str, parameter_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, parameter_name)

    def get_particle_mass_from_name(self, particle_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, 'mass')


class DotParser:
    def __init__(self, dot_path: str, model_path: str, verbose: bool = False):
        self.graph_file = pydot.graph_from_dot_file(dot_path)
        self.Model = ModelParser(model_path)
        self.verbose = verbose

    def get_dot_graph(self, process_id: int):
        return self.graph_file[str(process_id)]

    def infer_dependent_momentum(self,
                                 ext_momenta: list[list[float]],
                                 ext_sigs: list[int],
                                 dependent_momentum_index: int) -> list[float]:
        # Infering the dependent momentum from momentum conservation
        dmi = dependent_momentum_index
        dm_sig = ext_sigs[dmi]
        dependent_momentum = 4*[0.]
        exclusive_external_sigs = ext_sigs[:dmi] + ext_sigs[dmi+1:]
        for momentum, sig in zip(ext_momenta, exclusive_external_sigs):
            dependent_momentum[0] -= dm_sig*sig*momentum[0]
            dependent_momentum[1] -= dm_sig*sig*momentum[1]
            dependent_momentum[2] -= dm_sig*sig*momentum[2]
            dependent_momentum[3] -= dm_sig*sig*momentum[3]
        ext_momenta = ext_momenta[:dmi] + \
            [dependent_momentum] + ext_momenta[dmi:]

        if self.verbose:
            test_mom_cons = 4*[0.]
            for momentum, sig in zip(ext_momenta, ext_sigs):
                test_mom_cons[0] += sig*momentum[0]
                test_mom_cons[1] += sig*momentum[1]
                test_mom_cons[2] += sig*momentum[2]
                test_mom_cons[3] += sig*momentum[3]

            print("------------ INFERED EXTERNAL MOMENTUM --------------")
            print(f"{dependent_momentum=}")
            print(f"{test_mom_cons=} SHOULD BE ZERO")

        return ext_momenta

    def get_graph_properties(self, process_id: int,
                             ext_momenta: List[List[float]],
                             dependent_momentum_index: int) -> GraphProperties:
        from symbolica import E, S, Expression

        # External momenta
        n_ext_mom = len(ext_momenta) + 1
        ext_sigs = n_ext_mom*[0.]

        # Dot graph
        graph = self.get_dot_graph(process_id)
        edges = graph.get_edges()
        vertices = graph.get_nodes()

        VERTICES = []
        LMB_EDGES = []
        EXT_VERTICES = []
        INT_EDGES = []

        # Filter out the external vertices
        for vert in vertices:
            if vert.get('num') is not None:
                VERTICES.append(vert)

        # Add vertex ID for momtrop
        for v_id, vert in enumerate(VERTICES):
            vert.set('v_id', v_id)

        # Filter edges and add additional attributes for momtrop
        for edge in edges:
            src_split = edge.get_source().split(':')
            dst_split = edge.get_destination().split(':')
            edge.set('src', src_split[0])
            edge.set('dst', dst_split[0])

            if edge.get('lmb_id') is not None:
                LMB_EDGES.append(edge)

            if edge.get('source') is None:
                # Incoming external momentum
                EXT_VERTICES.append(graph.get_node(edge.get("dst"))[0])
                ext_sigs[int(dst_split[1])] = 1

            elif edge.get('sink') is None:
                # Outgoing external momentum
                EXT_VERTICES.append(graph.get_node(edge.get("src"))[0])
                ext_sigs[int(src_split[1])] = -1
            else:
                INT_EDGES.append(edge)
                particle_name = edge.get('particle')[1:-1]
                edge.set('mass', self.Model.get_particle_mass_from_name(
                    particle_name)[0])
                src_vert = graph.get_node(edge.get('src'))[0]
                dst_vert = graph.get_node(edge.get('dst'))[0]
                edge.set('src_id', src_vert.get('v_id'))
                edge.set('dst_id', dst_vert.get('v_id'))

        # Infer the missing external momentum
        ext_momenta = self.infer_dependent_momentum(
            ext_momenta, ext_sigs, dependent_momentum_index)

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S('P', 'K')
        x_, a_ = S('x_', 'a_')

        # Set up momtrop sampler
        TOLERANCE = 1E-10
        n_loops = len(LMB_EDGES)

        graph_externals = sorted([v.get("v_id") for v in EXT_VERTICES])
        graph_signature = []
        edge_momentum_shifts = []
        edge_src_dst_vertices = []
        edge_masses = []
        edge_ismassive = []

        for edge in INT_EDGES:
            # Generate the momtrop edge
            src_id = edge.get('src_id')
            dst_id = edge.get('dst_id')
            mass = edge.get('mass')
            edge_src_dst_vertices.append((src_id, dst_id))
            edge_masses.append(mass)
            edge_ismassive.append(mass > TOLERANCE)

            # LMB representation parsing
            e: Expression = E(edge.get('lmb_rep')[1:-1])
            e = e.replace(P(x_, a_), P(x_))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [int(e.coefficient(K(lmb_id)).to_sympy())
                       for lmb_id in range(n_loops)]
            graph_signature.append(lmb_sig)

            momentum_shift_sig = [float(e.coefficient(P(ext_id)).to_sympy())
                                  for ext_id in range(n_ext_mom)]
            momentum_shift = [0. for _ in range(3)]
            for coeff, ext_mom in zip(momentum_shift_sig, ext_momenta):
                for i in range(3):
                    momentum_shift[i] += coeff*ext_mom[i+1]

            if self.verbose:
                print(f"{momentum_shift_sig=}")
                print(f"{momentum_shift=}")

            edge_momentum_shifts.append(momentum_shift)

        if self.verbose:
            print("-------------- PARSED MOMTROP SAMPLER ---------------")
            print(f"{ext_momenta=}")
            print(f"{edge_masses=}")
            print(f"{graph_signature=}")
            print(f"{graph_externals=}")
            print(f"{edge_momentum_shifts=}")
            print(f"------------------ INTERNAL EDGES ------------------")
            for edge in INT_EDGES:
                print(edge.to_string())
            print(f"-------------------- LMB EDGES ---------------------")
            for edge in LMB_EDGES:
                print(edge.to_string())
            print(f"----------------- EXTERNAL VERTICES ----------------")
            for vert in EXT_VERTICES:
                print(vert.to_string())

        return GraphProperties(
            edge_src_dst_vertices=edge_src_dst_vertices,
            edge_masses=edge_masses,
            edge_ismassive=edge_ismassive,
            edge_momentum_shifts=edge_momentum_shifts,
            graph_external_vertices=graph_externals,
            graph_signature=graph_signature,
        )


class SettingsParser:
    def __init__(self, settings_path: str,
                 verbose: bool = False,
                 default_path: str = PATHS['default_settings'],):
        if not os.path.isabs(settings_path):
            settings_path = os.path.join(PATHS['tropnis'], settings_path)
        self.settings_path = settings_path
        self.verbose = verbose
        self.default_path = default_path
        with open(self.settings_path, 'r') as f:
            settings = toml.load(f)
            if settings is None:
                raise FileExistsError("The path to the settings file must be specified either relative "
                                      + "to the tropnis folder or be given as an absolute path.")
        with open(self.default_path, 'r') as f:
            default_settings = toml.load(f)
        self.settings = deep_update(default_settings, settings)
        self.gammaloop_state_path = os.path.join(PATHS['gammaloop_states'],
                                                 self.settings['gammaloop_state']['state_name'])
        self.gammaloop_runcard_path = os.path.join(self.gammaloop_state_path,
                                                   self.settings['gammaloop_state']['runcard_name'])
        self.dot_path = os.path.join(self.gammaloop_state_path,
                                     PATHS['state_to_dot'],
                                     self.settings['gammaloop_state']['process_name'],
                                     self.settings['gammaloop_state']['integrand_name']+".dot")
        self.model_path = os.path.join(self.gammaloop_state_path,
                                       self.settings['gammaloop_state']['model_name'])

    def get_gammaloop_integration_result(self):
        result_path = os.path.join(PATHS['gammaloop_states'],
                                   self.settings['gammaloop_state']['integration_state_name'],
                                   self.settings['gammaloop_state']['integration_result_file'])
        with open(result_path, 'r') as f:
            gammaloop_result = json.load(f)

        return gammaloop_result

    def get_model(self):
        return ModelParser(self.model_path)

    def get_gl_state(self) -> GammaLoopAPI:
        return GammaLoopAPI(self.gammaloop_state_path)

    def get_graph_properties(self) -> GraphProperties:
        Dot = DotParser(self.dot_path, self.model_path, self.verbose)
        ext_momenta, dmi = self.get_ext_momenta()
        graph_properties = Dot.get_graph_properties(
            self.settings['gammaloop_state']['process_id'], ext_momenta, dmi)
        n_int_edges = graph_properties.n_edges

        edge_weights = self.settings['momtrop']['edge_weight']
        match edge_weights:
            case int() | float():
                edge_weights = n_int_edges*[float(edge_weights)]
            case [_, *_]:
                if not len(edge_weights) == n_int_edges:
                    raise ValueError("If provided as a sequence, the number of momtrop "
                                     + "edgeweights must match the number of propagators.")
                edge_weights = edge_weights
            case "default":
                default_weight = (
                    3*graph_properties.n_loops + 3/2)/n_int_edges/2
                edge_weights = n_int_edges*[default_weight]
                if self.verbose:
                    print(
                        f"Setting momtrop edge weights to default: {default_weight:.5f}")
            case _:
                raise ValueError("Momtrop edge weights must be one of: \n"
                                 + "Number, Sequence of Numbers or \"default\".")

        graph_properties.momtrop_edge_weights = edge_weights

        return graph_properties

    def get_gl_integrand(self) -> Callable[[np.ndarray], np.ndarray]:
        def gammaloop_integrand(loop_momenta: np.ndarray,
                                state: GammaLoopAPI | None = None) -> np.ndarray:
            if state is None:
                state = self.get_gl_state()
            discrete_dims = np.zeros(
                (loop_momenta.shape[0], 1), dtype=np.uint64)
            res, jac = state.batched_inspect(
                points=loop_momenta, momentum_space=True,
                process_id=self.settings['gammaloop_state']['process_id'],
                integrand_name=self.settings['gammaloop_state']['integrand_name'],
                use_f128=False,  discrete_dims=discrete_dims
            )
            return res.real if self.settings['tropnis']['evaluate_real_part'] else res.imag

        return gammaloop_integrand

    def get_ext_momenta(self) -> Tuple[List[List[float]], int]:
        with open(self.gammaloop_runcard_path, 'r') as f:
            momenta_raw = f.read().split("momenta")[1].split("helicities")[0]
        before, after = momenta_raw.split("\"dependent\",")
        before = "list" + before + "]"
        after = "list = [" + after
        before, after = toml.loads(before)['list'], toml.loads(after)['list']
        dependent_momentum_index = len(before)
        momenta = before + after

        return momenta, dependent_momentum_index

    def get_momtrop_sampler(self, gp: GraphProperties
                            ) -> Tuple[momtrop.Sampler, momtrop.EdgeData, momtrop.Settings]:
        mt_edges = [momtrop.Edge(src_dst, ismassive, weight) for src_dst, ismassive, weight
                    in zip(gp.edge_src_dst_vertices, gp.edge_ismassive, gp.momtrop_edge_weights)]
        assym_graph = momtrop.Graph(mt_edges, gp.graph_external_vertices)
        momentum_shifts = [momtrop.Vector(*shift)
                           for shift in gp.edge_momentum_shifts]
        sampler = momtrop.Sampler(assym_graph, gp.graph_signature)
        edge_data = momtrop.EdgeData(gp.edge_masses, momentum_shifts)
        sampler_settings = momtrop.Settings(False, False)

        return sampler, edge_data, sampler_settings
