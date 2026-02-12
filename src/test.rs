use crate::vec3::Vec3;
pub type Point3 = Vec3;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let w = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(v + w, Vec3::new(5.0, 7.0, 9.0));
    }
    #[test]
    fn test_fn64mul() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(10.0 * v, Vec3::new(10.0, 20.0, 30.0));
    }
    #[test]
    fn test_ops() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let w = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(v + w, Vec3::new(5.0, 7.0, 9.0));

        assert_eq!(10.0 * v, Vec3::new(10.0, 20.0, 30.0));
    }
}
