import cairosvg
import csv
import datetime
import json
import os
import tempfile
import xlwt

from Products.Five.browser import BrowserView
from StringIO import StringIO
from zope.component import queryMultiAdapter


class ExportCSV(BrowserView):
    """ Export to CSV
    """

    def datapoints(self, response, chart_data):
        """ Export single dimension series to CSV
        """
        try:
            if len(chart_data) < 1:
                return ""
        except:
            return ""

        headers = ['series', 'name', 'code', 'y']

        keys = chart_data[0].get('data', [{}])[0].keys()

        response.write('Data extracted:\r\n')
        writer = csv.DictWriter(response, headers, restval='')
        writer.writeheader()

        for series in chart_data:
            for point in series['data']:
                encoded = {}
                encoded['series'] = series.get('name', '-')
                encoded['name'] = point.get('name', '-')
                for key in headers[1:]:
                    encoded[key] = unicode(point.get(key, '-')).encode('utf-8')
                    if point.get('isNA', False):
                        encoded['y'] = None
                writer.writerow(encoded)


    def datapoints_n(self, response, chart_data):
        """ Export multiple dimension series to CSV
        """
        try:
            if len(chart_data) < 1:
                return ""
        except:
            return ""

        coords = set(['x', 'y', 'z'])
        keys = set(chart_data[0][0].get('data', [{}])[0].keys())

        headers = ['series', 'name', 'x', 'y', 'z']

        if keys.intersection(coords) != coords:
            headers = ['series', 'name', 'x', 'y']

        writer = csv.DictWriter(response, headers, restval='')
        writer.writeheader()

        for series in chart_data:
            for point in series:
                encoded = {}
                encoded['series'] = point['name']
                for data in point['data']:
                    for key in headers[1:]:
                        encoded[key] = unicode(data[key]).encode('utf-8')
                    writer.writerow(encoded)


    def datapoints_profile(self, response, chart_data):
        headers = ['name', 'eu', 'original']
        extra_headers = ['period']

        writer = csv.DictWriter(response, extra_headers + headers, restval='')
        writer.writeheader()

        for series in chart_data:
            for point in series['data']:
                encoded = {}
                for key in headers:
                    encoded[key] = unicode(point[key]).encode('utf-8')
                period = point['attributes']['time-period']['notation']
                encoded['period'] = unicode(period).encode('utf-8')
                writer.writerow(encoded)


    def datapoints_profile_table(self, response, chart_data):
        for series in chart_data:
            encoded = {}
            latest = series['data']['latest']
            years = ['%s' % (latest-3), '%s' % (latest-2), '%s' % (latest-1), '%s' % (latest)]

            headers = (['country', 'indicator', 'breakdown', 'unit'] + years +
                       ['EU28 value %s' %latest, 'rank'])
            writer = csv.DictWriter(response, headers, restval='', dialect=csv.excel)
            writer.writeheader()

            encoded['country'] = series['data']['ref-area']['label']
            for ind in series['data']['table'].values():
                encoded['indicator'] = unicode(ind['indicator']).encode('utf-8')
                encoded['breakdown'] = unicode(ind['breakdown']).encode('utf-8')
                encoded['unit'] = unicode(ind['unit-measure']).encode('utf-8')
                for year in years:
                    encoded[year] = unicode(ind.get(year, '-')).encode('utf-8')
                #encoded['%s' %latest] = unicode(ind.get('%s' %latest, '-')).encode('utf-8')
                encoded['EU28 value %s' %latest] = unicode(
                        ind.get('eu', '-')).encode('utf-8')
                rank = ind.get('rank', '-')
                if rank == 0:
                    rank = '-'
                encoded['rank'] = unicode(rank).encode('utf-8')
                writer.writerow(encoded)


    def datapoints_profile_polar(self, response, chart_data):
        writer = csv.DictWriter(response, ['country', 'category', 'indicator', 'breakdown', 'unit', 'eu', 'original', 'period'], restval='')
        writer.writeheader()
        for series in chart_data:
            for point in series['data']:
                encoded = {}
                encoded['country'] = unicode(point['attributes']['ref-area']['notation']).encode('utf-8')
                encoded['category'] = unicode(point['title']).encode('utf-8')
                encoded['indicator'] = unicode(point['attributes']['indicator']['notation']).encode('utf-8')
                encoded['breakdown'] = unicode(point['attributes']['breakdown']['notation']).encode('utf-8')
                encoded['unit'] = unicode(point['attributes']['unit-measure']['notation']).encode('utf-8')
                encoded['eu'] = unicode(point['attributes']['eu']).encode('utf-8')
                encoded['original'] = unicode(point['attributes']['original']).encode('utf-8')
                encoded['period'] = unicode(point['attributes']['time-period']['notation']).encode('utf-8')
                writer.writerow(encoded)

    def write_metadata(self, response, metadata):
        writer = UnicodeWriter(response, dialect=csv.excel)
        writer.writerow(['Chart title:', metadata.get('chart-title', '-')])
        writer.writerow(['Source dataset:', metadata.get('source-dataset', '-')])
        writer.writerow([
            'Extraction-Date:',
            datetime.datetime.now().strftime('%d %b %Y')
        ])
        writer.writerow([
            'Link to the chart/table:',
            metadata.get('chart-url', '-')
        ])
        writer.writerow(['Selection of filters applied'])
        for item in metadata.get('filters-applied', []):
            writer.writerow(item)


    def write_annotations(self, response, annotations):
        writer = UnicodeWriter(response, dialect=csv.excel)
        writer.writerow([annotations.get('section_title', '-')])
        for item in annotations.get('blocks', []):
            writer.writerow([
                item.get('filter_label', '-') + ':',
                item.get('label', '-')
            ])
            if item.get('definition'):
                writer.writerow(['Definition:', item['definition']])
            if item.get('note'):
                writer.writerow(['Notes:', item['note']])
            if item.get('source_definition'):
                writer.writerow(['Source:', item['source_definition']])
        writer.writerow([
            'List of available indicators:',
            annotations.get('indicators_details_url')
        ])


    def export(self):
        """ Export to csv
        """

        to_xls = self.request.form.get('format')=='xls'

        if to_xls:
            self.request.response.setHeader(
                'Content-Type', 'application/vnd.ms-excel')
            self.request.response.setHeader(
                'Content-Disposition',
                'attachment; filename="%s.xls"' % self.context.getId())
        else:
            self.request.response.setHeader(
                'Content-Type', 'application/csv')
            self.request.response.setHeader(
                'Content-Disposition',
                'attachment; filename="%s.csv"' % self.context.getId())
        if not self.request.form.get('chart_data'):
            return
        chart_data = json.loads(self.request.form.pop('chart_data'))

        chart_type = self.request.form.pop('chart_type')

        metadata = {}
        if self.request.form.get('metadata'):
            metadata = json.loads(self.request.form.pop('metadata'))

        annotations = {}
        if self.request.form.get('annotations'):
            annotations = json.loads(self.request.form.pop('annotations'))

        formatters = {
            'scatter': self.datapoints_n,
            'bubbles': self.datapoints_n,
            'country_profile_bar': self.datapoints_profile,
            'country_profile_table': self.datapoints_profile_table,
            'country_profile_polar_polar': self.datapoints_profile_polar,
        }

        output_stream = self.request.response

        if to_xls:
            output_stream = StringIO()

        self.write_metadata(output_stream, metadata)

        formatter = formatters.get(chart_type, self.datapoints)
        formatter(output_stream, chart_data)

        self.write_annotations(output_stream, annotations)

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('chart data')

        if to_xls:
            output_stream.flush()
            output_stream.seek(0)
            source_csv = csv.reader(output_stream, delimiter=",")

            for rowi, row in enumerate(source_csv):
                for coli, value in enumerate(row):
                    sheet.write(rowi, coli, value.decode('utf-8'))
            with tempfile.TemporaryFile(mode='w+b') as f_temp:
                workbook.save(f_temp)
                f_temp.flush()
                f_temp.seek(0)
                chunk = True
                while chunk:
                    chunk = f_temp.read(64*1024)
                    self.request.response.write(chunk)
        return self.request.response

class ExportRDF(BrowserView):
    """ Export to RDF
    """
    def datapoints(self, points):
        """ xxx
        """
        return "Not implemented error"

    def datapoints_xy(self, points):
        """
        """
        return "Not implemented error"

    def export(self):
        """ Export to csv
        """
        options = self.request.form.get('options', "{}")
        method = self.request.form.get('method', 'datapoints')
        formatter = getattr(self, method, None)

        self.request.form = json.loads(options)
        points = queryMultiAdapter((self.context, self.request), name=method)

        self.request.response.setHeader(
            'Content-Type', 'application/rdf+xml')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="%s.rdf"' % self.context.getId())

        if not points:
            return ""

        if formatter:
            return formatter(points)

        return ""


def cairosvg_surface_color(string=None, opacity=1):
    """
    Replace ``string`` representing a color by a RGBA tuple.
    Function overwritten to catch exceptions at line 295.
    """
    from cairosvg.surface.colors import COLORS

    if not string or string in ("none", "transparent"):
        return (0, 0, 0, 0)

    string = string.strip().lower()

    if string in COLORS:
        string = COLORS[string]

    if string.startswith("rgba"):
        r, g, b, a = tuple(
            float(i.strip(" %")) * 2.55 if "%" in i else float(i)
            for i in string.strip(" rgba()").split(","))
        return r / 255, g / 255, b / 255, a * opacity
    elif string.startswith("rgb"):
        r, g, b = tuple(
            float(i.strip(" %")) / 100 if "%" in i else float(i) / 255
            for i in string.strip(" rgb()").split(","))
        return r, g, b, opacity

    if len(string) in (4, 5):
        string = "#" + "".join(2 * char for char in string[1:])
    if len(string) == 9:
        try:
            opacity *= int(string[7:9], 16) / 255
        except:
            pass

    try:
        plain_color = tuple(
            int(value, 16) / 255. for value in (
                string[1:3], string[3:5], string[5:7]))
    except ValueError:
        # Unknown color, return black
        return (0, 0, 0, 1)
    else:
        return plain_color + (opacity,)

class ExportSvg(BrowserView):
    def export(self):
        """
        Simply serve the png submitted as request parameter.
        This is needed only for the Content-Disposition header.
        """
        svg = self.request.get('svg')
        filename = self.request.get('filename', 'chart');
        self.request.response.setHeader(
            'Content-Type', 'image/svg+xml')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="' + filename + '.svg"')
        self.request.response.write(svg)
        return self.request.response

class SvgToPng(BrowserView):
    def convert(self):
        """
        Converts a svg to png and http returns the png.
        """
        svg = self.request.get('svg')
        filename = self.request.get('filename', 'chart');
        png_file = tempfile.TemporaryFile(mode='w+b')

        cairosvg.surface.color = cairosvg_surface_color
        cairosvg.svg2png(bytestring=svg, write_to=png_file)

        self.request.response.setHeader(
            'Content-Type', 'image/png')
        self.request.response.setHeader(
            'Content-Disposition',
            'attachment; filename="' + filename + '.png"')

        png_file.flush()
        png_file.seek(0)

        self.request.response.write(png_file.read())

        return self.request.response


class UnicodeWriter(object):
    def __init__(self, *args, **kwargs):
        self.writer = csv.writer(*args, **kwargs)

    def writerow(self, row):
        self.writer.writerow([
            x.encode('utf-8') if isinstance(x, unicode) else x for x in row])
