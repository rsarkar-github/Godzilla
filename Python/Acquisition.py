# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:04:30 2017
@author: rahul
"""
from Common import*
from CreateGeometry import CreateGeometry2D
from Utilities import TypeChecker


class Acquisition2D(object):
    """
    This class stores acquisition information
    """
    """
    TODO:
    1. Add exception handling
    2. Add plotting capabilities
    """
    def __init__(
            self,
            geometry2d=CreateGeometry2D()
    ):

        TypeChecker.check(x=geometry2d, expected_type=(CreateGeometry2D,))
        self.__geometry2D = geometry2d

        # Set default source and receiver positions
        self.__sources = {}
        self.__receivers = {}
        self.set_default_sources_receivers(source_skip=1, receiver_skip=1)

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        return self.__geometry2D == other.geometry2D \
            and self.__sources == other.sources \
            and self.__receivers == other.receivers

    def __ne__(self, other):

        if not isinstance(other, self.__class__):
            return True

        return self.__geometry2D != other.geometry2D \
            or self.__sources != other.sources \
            or self.__receivers != other.receivers

    def set_default_sources_receivers(self, source_skip=1, receiver_skip=1):

        TypeChecker.check_int_positive(x=source_skip)
        TypeChecker.check_int_positive(x=receiver_skip)

        self.__sources = {}
        self.__receivers = {}

        receivers = [
            (self.__geometry2D.ncellsX_pad + i, self.__geometry2D.ncellsZ_pad)
            for i in range(0, self.__geometry2D.ncellsX + 1, receiver_skip)
        ]
        for source_num, i in enumerate(range(0, self.__geometry2D.ncellsX + 1, source_skip)):
            self.__sources[source_num] = (self.__geometry2D.ncellsX_pad + i, self.__geometry2D.ncellsZ_pad)
            self.__receivers[source_num] = receivers

    def set_split_spread_acquisition(self, source_skip=1, receiver_skip=1, max_offset=None):

        TypeChecker.check_int_positive(x=source_skip)
        TypeChecker.check_int_positive(x=receiver_skip)

        self.__sources = {}
        self.__receivers = {}

        if max_offset is not None:
            TypeChecker.check_float_positive(x=max_offset)
        else:
            self.set_default_sources_receivers(source_skip=source_skip, receiver_skip=receiver_skip)
            return

        n = int(max_offset / self.__geometry2D.dx) + 1

        for source_num, i in enumerate(range(0, self.__geometry2D.ncellsX + 1, source_skip)):

            nx_source = self.__geometry2D.ncellsX_pad + i
            nz_source = self.__geometry2D.ncellsZ_pad

            nx_receiver_min = max(self.__geometry2D.ncellsX_pad, nx_source - n)
            nx_receiver_max = min(self.__geometry2D.ncellsX_pad + self.__geometry2D.ncellsX, nx_source + n)
            receivers = [
                (k, nz_source) for k in range(nx_receiver_min, nx_receiver_max + 1, receiver_skip)
            ]

            self.__sources[source_num] = (nx_source, nz_source)
            self.__receivers[source_num] = receivers

    def set_sources_receivers(self, sources, receivers):

        self.__sources = sources
        self.__receivers = receivers

    def crop_receivers_bounding_box(self, nx_start, nx_end, nz_start, nz_end):

        TypeChecker.check_int_bounds(
            x=nx_start,
            lb=self.__geometry2D.ncellsX_pad,
            ub=self.__geometry2D.ncellsX_pad + self.__geometry2D.ncellsX
        )

        TypeChecker.check_int_bounds(
            x=nx_end,
            lb=nx_start,
            ub=self.__geometry2D.ncellsX_pad + self.__geometry2D.ncellsX
        )

        TypeChecker.check_int_bounds(
            x=nz_start,
            lb=self.__geometry2D.ncellsZ_pad,
            ub=self.__geometry2D.ncellsZ_pad + self.__geometry2D.ncellsZ
        )

        TypeChecker.check_int_bounds(
            x=nz_end,
            lb=nz_start,
            ub=self.__geometry2D.ncellsZ_pad + self.__geometry2D.ncellsZ
        )

        sources = {}
        receivers = {}
        count = 0

        for source_num in self.__sources.keys():

            new_receivers = []
            for receiver in self.__receivers[source_num]:

                if nx_start <= receiver[0] <= nx_end and nz_start <= receiver[1] <= nz_end:
                    continue
                else:
                    new_receivers.append(receiver)

            if new_receivers:
                sources[count] = self.__sources[source_num]
                receivers[count] = new_receivers
                count += 1

        self.__sources = sources
        self.__receivers = receivers

    def crop_sources_bounding_box(self, nx_start, nx_end, nz_start, nz_end):

        TypeChecker.check_int_bounds(
            x=nx_start,
            lb=self.__geometry2D.ncellsX_pad,
            ub=self.__geometry2D.ncellsX_pad + self.__geometry2D.ncellsX
        )

        TypeChecker.check_int_bounds(
            x=nx_end,
            lb=nx_start,
            ub=self.__geometry2D.ncellsX_pad + self.__geometry2D.ncellsX
        )

        TypeChecker.check_int_bounds(
            x=nz_start,
            lb=self.__geometry2D.ncellsZ_pad,
            ub=self.__geometry2D.ncellsZ_pad + self.__geometry2D.ncellsZ
        )

        TypeChecker.check_int_bounds(
            x=nz_end,
            lb=nz_start,
            ub=self.__geometry2D.ncellsZ_pad + self.__geometry2D.ncellsZ
        )

        sources = {}
        receivers = {}
        count = 0

        for source_num in self.__sources.keys():

            if nx_start <= self.__sources[source_num][0] <= nx_end \
                    and nz_start <= self.__sources[source_num][1] <= nz_end:
                continue

            else:
                sources[count] = self.__sources[source_num]
                receivers[count] = self.__receivers[source_num]
                count += 1

        self.__sources = sources
        self.__receivers = receivers

    def crop_sources_receivers_bounding_box(self, nx_start, nx_end, nz_start, nz_end):

        self.crop_sources_bounding_box(nx_start=nx_start, nx_end=nx_end, nz_start=nz_start, nz_end=nz_end)
        self.crop_receivers_bounding_box(nx_start=nx_start, nx_end=nx_end, nz_start=nz_start, nz_end=nz_end)

    """
    # Properties
    """

    @property
    def geometry2D(self):

        return self.__geometry2D

    @geometry2D.setter
    def geometry2D(self, geometry2d):

        TypeChecker.check(x=geometry2d, expected_type=(CreateGeometry2D,))

        self.__geometry2D = geometry2d
        self.__sources = {}
        self.__receivers = {}
        self.set_default_sources_receivers(source_skip=1, receiver_skip=1)

    @property
    def sources(self):

        return self.__sources

    @property
    def receivers(self):

        return self.__receivers


if __name__ == "__main__":
    """
    Do something here
    """
