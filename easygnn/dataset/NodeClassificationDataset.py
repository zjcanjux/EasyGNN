import dgl
import dgl.function as fn
import torch as th
import numpy as np
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.data.utils import load_graphs, save_graphs
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset
from . import load_acm_raw
from . import BaseDataset, register_dataset
from . import AcademicDataset, HGBDataset, OHGBDataset
from .utils import sparse_mx_to_torch_sparse_tensor
from ..utils import add_reverse_edges





@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    r"""
    The class *NodeClassificationDataset* is a base class for datasets which can be used in task *node classification*.
    So its subclass should contain attributes such as graph, category, num_classes and so on.
    Besides, it should implement the functions *get_labels()* and *get_split()*.

    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The heterogeneous graph.
    category : str
        The category(or target) node type need to be predict. In general, we predict only one node type.
    num_classes : int
        The target node  will be classified into num_classes categories.
    has_feature : bool
        Whether the dataset has feature. Default ``False``.
    multi_label : bool
        Whether the node has multi label. Default ``False``. For now, only HGBn-IMDB has multi-label.
    """

    def __init__(self, *args, **kwargs):
        super(NodeClassificationDataset, self).__init__(*args, **kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict =None
        # self.in_dim = None

    def get_labels(self):
        r"""
        The subclass of dataset should overwrite the function. We can get labels of target nodes through it.

        Notes
        ------
        In general, the labels are th.LongTensor.
        But for multi-label dataset, they should be th.FloatTensor. Or it will raise
        RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 target' in call to _thnn_nll_loss_forward
        
        return
        -------
        labels : torch.Tensor
        """
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('Labels of nodes are not in the hg.nodes[category].data.')
        labels = labels.float() if self.multi_label else labels
        return labels

    def get_split(self, validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if 'train_mask' not in self.g.nodes[self.category].data:
            self.logger.dataset_info("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test
    
            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                random_int = th.randperm(len(train_idx))
                valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    # RDF_NodeClassification has train_mask, no val_mask
                    self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                    random_int = th.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        # Here set test_idx as attribute of dataset to save results of HGB
        return self.train_idx, self.valid_idx, self.test_idx




@register_dataset('hin_node_classification')
class HIN_NodeClassification(NodeClassificationDataset):
    r"""
    The HIN dataset are all used in different papers. So we preprocess them and store them as form of dgl.DGLHeteroGraph.
    The dataset name combined with paper name through 4(for).

    Dataset Name :
    acm4NSHE/ acm4GTN/ acm4NARS/ acm_han_raw/ academic4HetGNN/ dblp4MAGNN/ imdb4MAGNN/ ...
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(HIN_NodeClassification, self).__init__(*args, **kwargs)
        self.g, self.category, self.num_classes = self.load_HIN(dataset_name)

    def load_HIN(self, name_dataset):
        if name_dataset == 'demo_graph':
            data_path = './easygnn/dataset/demo_graph.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4NSHE':
            dataset = AcademicDataset(name='acm4NSHE', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'dblp4MAGNN':
            dataset = AcademicDataset(name='dblp4MAGNN', raw_dir='')
            category = 'A'
            g = dataset[0].long()
            num_classes = 4
            self.meta_paths_dict = {
                'APVPA': [('A', 'A-P', 'P'), ('P', 'P-V', 'V'), ('V', 'V-P', 'P'), ('P', 'P-A', 'A')],
                'APA': [('A', 'A-P', 'P'), ('P', 'P-A', 'A')],
            }
            self.meta_paths = [(('A', 'A-P', 'P'), ('P', 'P-V', 'V'), ('V', 'V-P', 'P'), ('P', 'P-A', 'A')),
                               (('A', 'A-P', 'P'), ('P', 'P-A', 'A'))]
            self.in_dim = g.ndata['h'][category].shape[1]

        elif name_dataset == 'imdb4MAGNN':
            dataset = AcademicDataset(name='imdb4MAGNN', raw_dir='')
            category = 'M'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'imdb4GTN':
            dataset = AcademicDataset(name='imdb4GTN', raw_dir='')
            category = 'movie'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4GTN':
            dataset = AcademicDataset(name='acm4GTN', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.meta_paths_dict = {'PAPSP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')],
                                    'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PSP': [('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')]
                                    }
            # self.meta_paths = [(('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
            #                     ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper'))]
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4NARS':
            dataset = AcademicDataset(name='acm4NARS', raw_dir='')
            g = dataset[0].long()
            num_classes = 3
            # g, labels, num_classes, train_nid, val_nid, test_nid = load_acm_nars()
            category = 'paper'
        elif name_dataset == 'acm4HeCo':
            dataset = AcademicDataset(name='acm4HeCo', raw_dir='')
            pos = sp.load_npz("./easygnn/dataset/acm4HeCo/pos.npz")
            self.pos = sparse_mx_to_torch_sparse_tensor(pos)
            g = dataset[0].long()
            num_classes = 3
            category = 'paper'
        elif name_dataset == 'academic4HetGNN':
            # which is used in HetGNN
            dataset = AcademicDataset(name='academic4HetGNN', raw_dir='')
            category = 'author'
            g = dataset[0].long()
            num_classes = 4
        elif name_dataset == 'yelp4HeGAN':
            # which is used in HeGAN
            dataset = AcademicDataset(name='yelp4HeGAN', raw_dir='')
            category = 'business'
            g = dataset[0].long()
            num_classes = 3
        elif name_dataset == 'HNE-PubMed':
            # which is used in HeGAN
            dataset = AcademicDataset(name='HNE-PubMed', raw_dir='')
            category = 'DISEASE'
            g = dataset[0].long()
            num_classes = 8
            g = add_reverse_edges(g)
            self.meta_paths_dict = {'DCD': [('DISEASE', 'CHEMICAL-in-DISEASE-rev', 'CHEMICAL'), ('CHEMICAL', 'CHEMICAL-in-DISEASE', 'DISEASE')],
                                    'DDD': [('DISEASE', 'DISEASE-and-DISEASE', 'DISEASE'), ('DISEASE', 'DISEASE-and-DISEASE-rev', 'DISEASE')],
                                    'DGD': [('DISEASE', 'GENE-causing-DISEASE-rev', 'GENE'), ('GENE', 'GENE-causing-DISEASE', 'DISEASE')],
                                    'DSD': [('DISEASE', 'SPECIES-with-DISEASE-rev', 'SPECIES'), ('SPECIES', 'SPECIES-with-DISEASE', 'DISEASE')]
                                    }
        elif name_dataset in ['acm_han', 'acm_han_raw']:
            if name_dataset == 'acm_han':
                pass
            elif name_dataset == 'acm_han_raw':
                g, category, num_classes, self.in_dim = load_acm_raw(False)
            else:
                return NotImplementedError('Unsupported dataset {}'.format(name_dataset))
            return g, category, num_classes
        elif name_dataset in ['demo']:
            data_path = './easygnn/dataset/graph.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            self.in_dim = g.ndata['h'][category].shape[1]
        # g, _ = load_graphs(data_path)
        # g = g[0]
        return g, category, num_classes