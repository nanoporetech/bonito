PACKAGE  ?= ont-bonito
MAJOR    ?= $(shell awk -F"['.]" '/^__version__/{print $$2}' bonito/__init__.py)
MINOR    ?= $(shell awk -F"['.]" '/^__version__/{print $$3}' bonito/__init__.py)
SUB      ?= $(shell awk -F"['.]" '/^__version__/{print $$4}' bonito/__init__.py)
PATCH    ?= 1
CODENAME ?= $(shell awk -F= '/CODENAME/{print $$2}' /etc/lsb-release)
VERSION   = $(MAJOR).$(MINOR).$(SUB)-$(PATCH)

INSTALL_DIR  := /opt/ont/bonito
INSTALL_VENV := $(INSTALL_DIR)/venv/
TORCH_VERSION = 1.1.0

define DEB_CONTROL
Package: $(PACKAGE)
Version: $(MAJOR).$(MINOR).$(SUB)-$(PATCH)~$(CODENAME)
Priority: optional
Section: science
Architecture: amd64
Depends: python3
Maintainer: Chris Seymour <chris.seymour@nanoporetech.com>
Description: Bonito Basecaller
endef
export DEB_CONTROL

clean:
	rm -rf build deb_dist dist bonito*.tar.gz bonito.egg-info *~ *.deb archive
	find . -name "*~" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

deb: clean
	touch tmp
	rm -rf tmp

	mkdir -p tmp/DEBIAN
	echo "$$DEB_CONTROL" > tmp/DEBIAN/control
	echo "#!/bin/bash\nln -fs $(INSTALL_VENV)/bin/bonito /usr/local/bin" > tmp/DEBIAN/postinst
	echo "#!/bin/bash\nrm -f /usr/local/bin/bonito" > tmp/DEBIAN/postrm

	mkdir -p tmp$(INSTALL_DIR)

        # setup virtualenv
	python3 -m venv --copies tmp$(INSTALL_VENV)
	. tmp$(INSTALL_VENV)bin/activate; pip install --upgrade pip
	. tmp$(INSTALL_VENV)bin/activate; pip install torch==$(TORCH_VERSION)
	. tmp$(INSTALL_VENV)bin/activate; pip install -r requirements.txt
        # install bonito
	python3 setup.py sdist
	. tmp$(INSTALL_VENV)bin/activate; pip install dist/$(PACKAGE)-$(MAJOR).$(MINOR).$(SUB).tar.gz
        # update scripts shebang
	sed -i "1s%.*%#!${INSTALL_VENV}bin/python3%" tmp$(INSTALL_VENV)/bin/bonito

	find tmp -type f ! -regex '.*\(\bDEBIAN\b\|\bdeb-src\b\).*' -exec md5sum {} \; | sed 's%tmp/%%' > tmp/DEBIAN/md5sums

	chmod 644 tmp/DEBIAN/control
	chmod 644 tmp/DEBIAN/md5sums
	chmod 755 tmp/DEBIAN/postinst
	chmod 755 tmp/DEBIAN/postrm

        # package everything up
	(cd tmp; fakeroot dpkg -b . ../$(PACKAGE)_$(MAJOR).$(MINOR).$(SUB)-$(PATCH).deb)
	rm -rf tmp
