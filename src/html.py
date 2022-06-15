import os
import shutil
import numpy as np

from mako.template import Template
from mako.lookup import TemplateLookup


def make_simple_index(path, path_list):
    lookup = TemplateLookup(directories=['templates/', '.'])
    t = Template(filename=os.path.join('templates', 'simple.html'),
                 lookup=lookup)
    with open(os.path.join(path, 'index.html'), 'w') as f:
        f.write(t.render(paths=path_list))
    assets = os.path.join(path, 'assets')
    if not os.path.exists(assets):
        os.makedirs(assets)
    shutil.copyfile(os.path.join('templates', 'bulma-slider.min.js'),
                    os.path.join(assets, 'bulma-slider.min.js'))
    shutil.copyfile(os.path.join('templates', 'fabric.min.js'),
                    os.path.join(assets, 'fabric.min.js'))
    shutil.copyfile(os.path.join('templates', 'bulma-slider.min.css'),
                    os.path.join(assets, 'bulma-slider.min.css'))
    shutil.copyfile(os.path.join('templates', 'bulma.min.css'),
                    os.path.join(assets, 'bulma.min.css'))


def make_list_index(path, samples, steps):
    lookup = TemplateLookup(directories=['templates/', '.'])
    t = Template(filename=os.path.join('templates', 'list.html'),
                 lookup=lookup)
    with open(os.path.join(path, 'index.html'), 'w') as f:
        f.write(t.render(ids=list(range(samples)),last_idx=steps-1))
    assets = os.path.join(path, 'assets')
    if not os.path.exists(assets):
        os.makedirs(assets)
    shutil.copyfile(os.path.join('templates', 'bulma-slider.min.js'),
                    os.path.join(assets, 'bulma-slider.min.js'))
    shutil.copyfile(os.path.join('templates', 'fabric.min.js'),
                    os.path.join(assets, 'fabric.min.js'))
    shutil.copyfile(os.path.join('templates', 'bulma-slider.min.css'),
                    os.path.join(assets, 'bulma-slider.min.css'))
    shutil.copyfile(os.path.join('templates', 'bulma.min.css'),
                    os.path.join(assets, 'bulma.min.css'))
    shutil.copyfile(os.path.join('templates', 'bulma-accordion.min.css'),
                    os.path.join(assets, 'bulma-accordion.min.css'))
    shutil.copyfile(os.path.join('templates', 'bulma-accordion.min.js'),
                    os.path.join(assets, 'bulma-accordion.min.js'))

