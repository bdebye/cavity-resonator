# Copyright (C) 2025 Wen Wang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


class gmsh_post(object):

    def __init__(self, filename):
        self.file = open(filename, "w")
        self.view_open = False

    def open_view(self, viewname):
        self.file.write(f"View \"{viewname}\" {{\n")
        self.view_open = True

    def add_scalar_field(self, coords, a):
        assert self.view_open
        template = "\tSP ( %f, %f, %f ){ %e };\n"
        self.file.write(template % (*coords, a))

    def add_vector_field(self, coords, v):
        assert self.view_open
        template = "\tVP ( %f, %f, %f ){ %e, %e, %e };\n"
        self.file.write(template % (*coords, *v))

    def close_vew(self):
        self.file.write("};\n")
        self.view_open = False

    def close(self):
        if self.view_open:
            self.close_vew()
        self.file.close()

    def __del__(self):
        if self.view_open:
            self.file.write("};\n")
            self.view_open = False

        if not self.file.closed:
            self.file.close()
