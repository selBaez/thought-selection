/*global jQuery */


$("#subject_label").change(function () {
    var subjectArray = ['corazao', $("#subject_label").val()];
    $("#subject_uri").text(addressArray.join(' '));
});