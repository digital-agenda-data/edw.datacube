import csv
import simplejson as json
from Products.Five.browser import BrowserView

from eea.cache import cache as eeacache


def jsonify(request, data, cache=True):
    header = request.RESPONSE.setHeader
    header("Content-Type", "application/json")
    if cache:
        header("Expires", "Sun, 17-Jan-2038 19:14:07 GMT")
    return json.dumps(data, indent=2, sort_keys=True)

def cacheKey(method, self, *args, **kwargs):
    """ Generate unique cache id
    """
    return (self.context.absolute_url(1), dict(self.request.form))

class AjaxDataView(BrowserView):

    def __init__(self, context, request):
        super(AjaxDataView, self).__init__(context, request)
        self.endpoint = self.request.get('endpoint', '')
        self.cube = context.get_cube(self.endpoint)

    def jsonify(self, *args, **kwargs):
        return jsonify(self.request, *args, **kwargs)

    def datasets(self):
        return self.jsonify({'datasets': self.cube.get_datasets()},
                            cache=False)

    def dataset_metadata(self):
        dataset = self.request.form['dataset']
        return self.jsonify(self.cube.get_dataset_metadata(dataset))

    def dataset_details(self):
        return self.jsonify(self.cube.get_dataset_details())

    def get_dimensions(self):
        flat = bool(self.request.form.get('flat'))
        dimensions = self.cube.get_dimensions(flat=flat)
        return self.jsonify(dimensions)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_labels(self):
        form = dict(self.request.form)
        dimension = form.pop('dimension')
        value = form.pop('value')
        [labels] = self.cube.get_dimension_labels(dimension, value)
        return self.jsonify(labels)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        filters = sorted(form.items())
        options = self.cube.get_dimension_options(dimension, filters)
        return self.jsonify({'options': options})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options_xy(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        (filters, x_filters, y_filters) = ([], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        options = self.cube.get_dimension_options_xy(dimension, filters,
                                                     x_filters, y_filters)
        return self.jsonify({'options': options})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def dimension_options_xyz(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        dimension = form.pop('dimension')
        (filters, x_filters, y_filters, z_filters) = ([], [], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            elif k.startswith('z-'):
                z_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        options = self.cube.get_dimension_options_xyz(dimension, filters,
                                                      x_filters, y_filters,
                                                      z_filters)
        return self.jsonify({'options': options})

    def dimension_value_metadata(self):
        dimension = self.request.form['dimension']
        value = self.request.form['value']
        res = self.cube.get_dimension_option_metadata(dimension, value)
        return self.jsonify(res)

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        columns = form.pop('columns', form.pop('fields', '')).split(',')
        filters = sorted(form.items())
        rows = list(self.cube.get_observations(filters=filters))
        return self.jsonify({'datapoints': rows})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints_xy(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        columns = form.pop('columns').split(',')
        xy_columns = form.pop('xy_columns').split(',')
        (filters, x_filters, y_filters) = ([], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        rows = list(self.cube.get_data_xy(columns=columns,
                                          xy_columns=xy_columns,
                                          filters=filters,
                                          x_filters=x_filters,
                                          y_filters=y_filters))
        return self.jsonify({'datapoints': rows})

    @eeacache(cacheKey, dependencies=['edw.datacube'])
    def datapoints_xyz(self):
        form = dict(self.request.form)
        form.pop('rev', None)
        columns = form.pop('columns').split(',')
        xyz_columns = form.pop('xyz_columns').split(',')
        (filters, x_filters, y_filters, z_filters) = ([], [], [], [])
        for k, v in sorted(form.items()):
            if k.startswith('x-'):
                x_filters.append((k[2:], v))
            elif k.startswith('y-'):
                y_filters.append((k[2:], v))
            elif k.startswith('z-'):
                z_filters.append((k[2:], v))
            else:
                filters.append((k, v))
        rows = list(self.cube.get_data_xyz(columns=columns,
                                          xyz_columns=xyz_columns,
                                          filters=filters,
                                          x_filters=x_filters,
                                          y_filters=y_filters,
                                          z_filters=z_filters))
        return self.jsonify({'datapoints': rows})

    def dump_csv(self):
        response = self.request.response
        response.setHeader('Content-type', 'text/csv; charset=utf-8')
        filename = self.context.getId() + '.csv'
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        headers = [
            'indicator',
            'breakdown',
            'unit_measure',
            'time_period',
            'ref_area',
            'value']
        writer = csv.DictWriter(response, headers, restval='')
        writer.writeheader()
        for row in self.cube.dump():
            encoded_row = {}
            for k,v in row.iteritems():
                encoded_row[k] = unicode(v).encode('utf-8')
            writer.writerow(encoded_row)
        return response

    def dump_rdf(self):
        response = self.request.response
        response.setHeader('Content-type', 'application/rdf+xml; charset=utf-8')
        filename = self.context.getId() + '.rdf'
        response.setHeader('Content-Disposition',
                           'attachment;filename=%s' % filename)
        return self.cube.dump(data_format='application/rdf+xml')
