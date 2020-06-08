var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.send = send;

var _CoreManager = _interopRequireDefault(require("./CoreManager"));

var _ParseQuery = _interopRequireDefault(require("./ParseQuery"));

function send(data, options) {
  options = options || {};

  if (data.where && data.where instanceof _ParseQuery.default) {
    data.where = data.where.toJSON().where;
  }

  if (data.push_time && typeof data.push_time === 'object') {
    data.push_time = data.push_time.toJSON();
  }

  if (data.expiration_time && typeof data.expiration_time === 'object') {
    data.expiration_time = data.expiration_time.toJSON();
  }

  if (data.expiration_time && data.expiration_interval) {
    throw new Error('expiration_time and expiration_interval cannot both be set.');
  }

  return _CoreManager.default.getPushController().send(data, {
    useMasterKey: options.useMasterKey
  });
}

var DefaultController = {
  send: function (data, options) {
    var RESTController = _CoreManager.default.getRESTController();

    var request = RESTController.request('POST', 'push', data, {
      useMasterKey: !!options.useMasterKey
    });
    return request;
  }
};

_CoreManager.default.setPushController(DefaultController);