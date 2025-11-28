"""
Use these ROI masks to extract traces.
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field


class RoiMaskDataTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # ROI mask
        -> self.field_table
        -> self.raw_params_table
        ---
        -> self.presentation_table
        roi_mask     : blob                   # ROI mask for recording field
        """
        return definition

    class RoiMaskPresentation(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> master
            -> self.presentation_table
            ---
            roi_mask      : blob       # ROI mask for presentation field
            as_field_mask : enum("same", "different", "shifted")  # relationship to field mask
            shift_dx=0    : int  # Shift in x
            shift_dy=0    : int  # Shift in y
            """
            return definition

        @property
        @abstractmethod
        def presentation_table(self):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    def plot1(self, key=None, gamma=0.7):
        key = get_primary_key(table=self.proj() * self.presentation_table.proj(), key=key)
        npixartifact, scan_type = (self.field_table & key).fetch1("npixartifact", "scan_type")
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        try:
            alt_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1(
                'ch_average')
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)

        roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
        plot_field(main_ch_average, alt_ch_average, scan_type=scan_type,
                   roi_mask=roi_mask, title=key, npixartifact=npixartifact,
                   gamma=gamma)
