import numpy as np
import ntpath
import os
import sys
import time
import visdom

from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """Used to display training images and results live."""
    def __init__(self, opt):
        self.saved = False
        self.name = opt.dataset_name
    
        # Display
        self.display_id = opt.display_id
        self.env = opt.display_env
        self.server = opt.display_server
        self.port = opt.display_port
        self.win_size = opt.display_winsize

        self.web_dir = os.path.join(opt.checkpoints_dir, self.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        util.mkdirs([self.web_dir, self.img_dir])

        # Visdom object
        self.vis = visdom.Visdom(server=self.server, port=self.port, env=self.env)

        if not self.vis.check_connection():
            self.create_visdom_connections()

    def reset(self):
        """Reset saved to False, ensuring images are saved."""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        ncols = len(visuals)  # Number of images to display per row

        h, w = next(iter(visuals.values())).shape[:2]
        table_css = """<style>
                table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                </style>""" % (w, h)  # create a table css
        # create a table of images.
        title = self.name
        label_html = ''
        label_html_row = ''
        images = []
        idx = 0
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image * 255) # Denormalize data
            label_html_row += '<td>%s</td>' % label
            # TODO(igorruvinov) figure out how to get images to display next to each other without reshaping
            images.append(image_numpy.reshape([1, image_numpy.shape[0], image_numpy.shape[1]]))
            idx += 1
            if idx % ncols == 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
        white_image = np.ones_like(image_numpy) * 255
        while idx % ncols != 0:
            images.append(white_image)
            label_html_row += '<td></td>'
            idx += 1
        if label_html_row != '':
            label_html += '<tr>%s</tr>' % label_html_row
        try:
            self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                            padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=self.display_id + 2,
                            opts=dict(title=title + ' labels'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

        if save_result or not self.saved:  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image * 255) # Denormalize data
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
