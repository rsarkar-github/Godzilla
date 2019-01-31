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

        receivers = [
            (self.__geometry2D.ncellsX_pad + i, self.__geometry2D.ncellsZ_pad)
            for i in range(0, self.__geometry2D.ncellsX + 1, receiver_skip)
        ]
        for source_num, i in enumerate(range(0, self.__geometry2D.ncellsX + 1, source_skip)):
            self.__sources[source_num] = (self.__geometry2D.ncellsX_pad + i, self.__geometry2D.ncellsZ_pad)
            self.__receivers[source_num] = receivers

    def set_sources_receivers(self, sources, receivers):

        self.__sources = sources
        self.__receivers = receivers

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
