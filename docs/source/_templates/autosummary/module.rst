{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

.. currentmodule:: romtools

{% for heading, items in [('Functions', functions), ('Classes', classes), ('Exceptions', exceptions)] %}
{% if items %}
.. rubric:: {{ heading }}

.. autosummary::
   :toctree:
{% for item in items %}
   {{ public_api_aliases.get(fullname + '.' + item, fullname + '.' + item)[9:] }}
{% endfor %}
{% endif %}
{% endfor %}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
{% for item in attributes %}
   {{ public_api_aliases.get(fullname + '.' + item, fullname + '.' + item)[9:] }}
{% endfor %}
{% endif %}

{# Submodules are listed explicitly in api.rst, never recursively discovered. #}
