# type: ignore
from dataclasses import dataclass
import momtrop
import pydot
import json
import toml
import os

from symbolica import E, S, Expression


@dataclass
class MomtropSamplerProperties:
    edge_weights: list[int]
    edge_masses: list[int]
    edges: list[momtrop.Edge]
    external_vertices: list[int]
    signature: list[list[int]]
    momentum_shifts: list[momtrop.Vector]


class RunCardParser:
    def __init__(self, runcard_path: str):
        self.runcard_path = runcard_path
        with open(self.runcard_path, 'r') as f:
            self.runcard = toml.load(f)
            self.tropnis_settings = self.runcard['tropnis_settings']
        self.dot_path = self.tropnis_settings['processed_dot_file']
        self.model_path = self.tropnis_settings['model_json_file']
        self.dot_graph = pydot.graph_from_dot_file(self.dot_path)[0]
        with open(self.model_path, 'r') as f:
            self.model = json.load(f)

    def get_particle_parameter(self, particle_name: str, parameter_name: str):
        particle_match = None
        for particle in self.model['particles']:
            try:
                if particle['name'] == particle_name:
                    particle_match = particle
            except:
                pass

        if particle_match is None:
            raise KeyError(
                f"Particle '{particle_name}' does not exist in model '{self.model_path}'.")

        try:
            model_parameter_name = particle_match[parameter_name]
        except KeyError:
            raise KeyError(
                f"Particle '{particle_name}' does not have parameter '{parameter_name}'.")

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
            raise KeyError(f"The model '{self.model_path}' does not specify a value for the \
                           parameter '{model_parameter_name}'. This should not happen!!!")

        return parameter_match

    def generate_momtrop_sampler_from_dot_file(self) -> tuple[momtrop.Sampler, MomtropSamplerProperties]:
        # External momenta
        ext_momenta = self.runcard['default_runtime_settings']['kinematics']['externals']['data']['momenta']
        # Import the dot graph
        graph = self.dot_graph
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
            edge.set('src', edge.get_source().split(':')[0])
            edge.set('dst', edge.get_destination().split(':')[0])

            if edge.get('lmb_id') is not None:
                LMB_EDGES.append(edge)

            if edge.get('source') is None:
                EXT_VERTICES.append(graph.get_node(edge.get("dst"))[0])
            elif edge.get('sink') is None:
                EXT_VERTICES.append(graph.get_node(edge.get("src"))[0])
            else:
                INT_EDGES.append(edge)
                particle = edge.get('particle')[1:-1]
                edge.set('mass', self.get_particle_property(
                    particle, 'mass')[0])
                src_vert = graph.get_node(edge.get('src'))[0]
                dst_vert = graph.get_node(edge.get('dst'))[0]
                edge.set('src_id', src_vert.get('v_id'))
                edge.set('dst_id', dst_vert.get('v_id'))

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S('P', 'K')
        x_, a_ = S('x_', 'a_')

        # Set up momtrop sampler
        TOLERANCE = 1E-10
        n_loops = len(LMB_EDGES)
        n_int = len(INT_EDGES)
        n_ext = len(edges) - n_int
        mt_weight = (3*n_loops + 3/2)/n_int/2

        mt_edges = []
        mt_masses = []
        mt_signature = []
        mt_offsets = []
        mt_externals = sorted([v.get("v_id") for v in EXT_VERTICES])

        for edge in INT_EDGES:
            # Generate the momtrop edge
            src_id = edge.get('src_id')
            dst_id = edge.get('dst_id')
            mass = edge.get('mass')
            mt_edges.append(
                momtrop.Edge((src_id, dst_id), mass > TOLERANCE, mt_weight)
            )
            mt_masses.append(mass)

            # LMB representation parsing
            e: Expression = E(edge.get('lmb_rep')[1:-1])
            e = e.replace(P(x_, a_), P(x_-1))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [int(e.coefficient(K(lmb_id)).to_sympy())
                       for lmb_id in range(n_loops)]
            mt_signature.append(lmb_sig)

            offset_sig = [float(e.coefficient(P(ext_id)).to_sympy())
                          for ext_id in range(n_ext - 1)]
            offset = [0. for _ in range(3)]
            for coeff, ext_mom in zip(offset_sig, ext_momenta):
                for i in range(3):
                    offset[i] += coeff*ext_mom[i+1]

            mt_offsets.append(momtrop.Vector(*offset))

        assym_graph = momtrop.Graph(mt_edges, mt_externals)
        sampler = momtrop.Sampler(assym_graph, mt_signature)

        sampler_properties = MomtropSamplerProperties(
            edge_weights=mt_weight,
            edge_masses=mt_masses,
            edges=mt_edges,
            external_vertices=mt_externals,
            signature=mt_signature,
            momentum_shifts=mt_offsets,
        )

        return sampler, sampler_properties
